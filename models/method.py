import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_backbones.resnet18 import ResNet18
from models.resnet12 import ResNet12
from models.scr import SCR, SelfCorrelationComputation
from models.cats import TransformerAggregator
from functools import reduce, partial
from models.mod import FeatureL2Norm
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
import sys
from vit_pytorch import ViT
from common.utils import *
from models.feature_backbones.vision_transformer import vit_small
import torch.nn.init as init


class Method(nn.Module):

    def __init__(self,
                 args,
                 mode=None,
                 feature_size=8,
                 feature_proj_dim=8,
                 depth=1,
                 num_heads=2,
                 mlp_ratio=4):
        super().__init__()
        self.mode = mode
        self.args = args

        vit_dim = 384
        self.vit_dim = vit_dim
        hyperpixel_ids = args.hyperpixel_ids
        self.encoder = vit_small(patch_size=16, return_all_tokens=True)
        chkpt = torch.load(f"/home/chenderong/work/MCNet/checkpoints/{args.dataset}/vit_weight/checkpoint1600.pth")
        chkpt_state_dict = chkpt['teacher']
        self.encoder.load_state_dict(match_statedict(chkpt_state_dict), strict=False)
        self.classification_head = nn.Linear(384, self.vit_dim)
        self.encoder_dim = vit_dim
        self.hyperpixel_ids = hyperpixel_ids
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        self.args = args
        self.proj = nn.Linear(vit_dim, feature_proj_dim)
        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))

        self.l2norm = FeatureL2Norm()

        init.xavier_uniform_(self.classification_head.weight)
        init.xavier_uniform_(self.proj.weight)
        init.xavier_uniform_(self.fc.weight)

    def corr(self, src, trg):
        outer_product_matrix = torch.bmm(src.unsqueeze(2), trg.unsqueeze(1))
        # 去除多余的维度
        return outer_product_matrix.squeeze()

    def get_correlation_map(self, spt, qry, idx, num_qry, way):

        # [75x25,1,5,5]

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        spt = spt.view(num_qry, way, *spt.size()[1:])
        qry = qry.view(num_qry, way, *qry.size()[1:])
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        _, _, h, w, _, _ = similarity_map_einsum.size()
        similarity_map = similarity_map_einsum.view(-1, h * w, h * w)

        return similarity_map

    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (3, 3), 1
        layers = list()

        if self.args.self_method == 'scr':
            corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            self_block = SCR(planes=planes, stride=stride)
        else:
            raise NotImplementedError

        if self.args.self_method == 'scr':
            layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = self.fc(x)
        return x

    def cca(self, spt, qry):

        # shifting channel activations by the channel mean
        # shape of spt : [25, 9]
        # shape of qry : [75, 9]
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # ----------------------------------cat--------------------------------------#

        # spt_feats = spt.unsqueeze(0).repeat(num_qry, 1, 1).view(-1,*spt.size()[1:]) #shape of spt_feats [75x25, 9]
        # qry_feats = qry.unsqueeze(1).repeat(1, way, 1).view(-1,*qry.size()[1:]) #[75x25, 9]
        #
        # corr = self.corr(spt_feats, qry_feats).unsqueeze(1).repeat(1,1,1,1) #the shape of corr : [75x25,2, 9, 9]
        # spt_feats_proj = self.proj(spt_feats).unsqueeze(1).unsqueeze(2).repeat(1, 1, self.vit_dim, 1) #[75x25,2,9,3]
        # qry_feats_proj = self.proj(qry_feats).unsqueeze(1).unsqueeze(2).repeat(1, 1, self.vit_dim, 1) #[75x25,2,9,3]
        #
        # refined_corr = self.decoder(corr, spt_feats_proj, qry_feats_proj).view(num_qry,way,*[self.feature_size]*4)
        # corr_s = refined_corr.view(num_qry, way, self.feature_size*self.feature_size, self.feature_size*self.feature_size)
        # corr_q = refined_corr.view(num_qry, way, self.feature_size*self.feature_size, self.feature_size*self.feature_size)
        #
        # # applying softmax for each side
        # corr_s = F.softmax(corr_s / self.args.temperature_attn, dim=2)
        # corr_q = F.softmax(corr_q / self.args.temperature_attn, dim=3)
        #
        # # suming up matching scores
        # attn_s = corr_s.sum(dim=[3])
        # attn_q = corr_q.sum(dim=[2])
        #
        # # applying attention
        # spt_attended = attn_s * spt_feats.view(num_qry, way, *spt_feats.shape[1:]) #[75, 25, 9]
        # qry_attended = attn_q * qry_feats.view(num_qry, way, *qry_feats.shape[1:]) #[75, 25, 9]

        # ----------------------------------cat--------------------------------------#

        spt_attended = spt.unsqueeze(0).repeat(num_qry, 1, 1)
        qry_attended = qry.unsqueeze(1).repeat(1, way, 1)

        # ----------------------------------replace--------------------------------------#

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        similarity_matrix = F.cosine_similarity(spt_attended, qry_attended, dim=-1)
        # similarity_matrix = -F.pairwise_distance(spt_attended_pooled.view(num_qry*self.args.way,-1), qry_attended_pooled.view(num_qry*self.args.way,-1), p=2).view(num_qry,self.args.way)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry)
        else:
            return similarity_matrix / self.args.temperature

    def visualize(self, t1, t2):

        t1 = t1.view(75 * 5, 1024).cpu().detach()
        t = t1[0].unsqueeze(0)
        offset = 1
        for i in range(5, 75 * 5, 5):
            t = torch.cat([t, t1[i + offset].unsqueeze(0)], dim=0)
            offset = (offset + 1) % 5
        label1 = [1, 2, 3, 4, 5] * 15

        t2 = t2.view(75 * 5, 1024).cpu().detach()
        offset = 0
        for i in range(0, 75 * 5, 5):
            t = torch.cat([t, t2[i + offset].unsqueeze(0)], dim=0)
            offset = (offset + 1) % 5
        label2 = [6, 7, 8, 9, 10] * 15

        label = label1 + label2

        ts = TSNE(n_components=2, learning_rate="auto", init="pca", random_state=33)
        result = ts.fit_transform(t)
        self.plot_embedding(result, label, '')

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x):
        feats = self.encoder(x)[:,0,:]
        # feats = self.classification_head(feats)
        return feats

    def plot_embedding(self, data, label, title):
        # plt.rc('font',family='Times New Roman')
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
        fig = plt.figure()  # 创建图形实例
        ax = plt.subplot(111)  # 创建子图
        # 遍历所有样本

        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], '.' if label[i] <= 5 else 'X', color=plt.cm.Set1(label[i] % 5),
                     fontdict={'weight': 'bold', 'size': 28 if label[i] <= 5 else 14})
        plt.xticks(fontproperties="Times New Roman")  # 指定坐标的刻度
        plt.yticks(fontproperties="Times New Roman")
        # plt.xticks()		# 指定坐标的刻度
        # plt.yticks()
        plt.title("Distribution of task" + str(self.args.visualfile) + " generated by ML", fontsize=14)
        plt.savefig("/data/data-home/chenderong/work/MCNet/visualizes/" + str(self.args.visualfile) + ".pdf",
                    format="pdf")
        # 返回值
        return fig
