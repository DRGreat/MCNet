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


class Method(nn.Module):

    def __init__(self, 
    args,  
    mode=None,
    feature_size=5,
    feature_proj_dim=1,
    depth=1,
    num_heads=2,
    mlp_ratio=4):
        super().__init__()
        self.mode = mode
        self.args = args

        channels =  [1]  + [1]  + [1]  + [1]
        hyperpixel_ids = args.hyperpixel_ids
        self.encoder = ViT(
            image_size = 84,
            patch_size = 14,
            num_classes = 25,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.encoder_dim = sum([channels[i] for i in hyperpixel_ids])
        self.channels = channels
        self.hyperpixel_ids = hyperpixel_ids
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim
        self.args = args
        
        self.proj = nn.ModuleList([
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])
        self.cca_1x1 = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(channels[i], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()) for i in hyperpixel_ids
        ])
        self.scr_module = nn.ModuleList([
            self._make_scr_layer(planes=[channels[i], 64, 64, 64, channels[i]]) for i in hyperpixel_ids
            ])
        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids))
            
        self.l2norm = FeatureL2Norm()


    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)
    
    def get_correlation_map(self, spt, qry,idx,num_qry,way):

         # [75x25,1,5,5]

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

       
        spt = spt.view(num_qry, way, *spt.size()[1:])
        qry = qry.view(num_qry, way, *qry.size()[1:])
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        _,_,h,w,_,_ = similarity_map_einsum.size()
        similarity_map = similarity_map_einsum.view(-1, h*w,h*w)
        
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
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        
        spt = spt.squeeze(0) #shape of spt : [25, 1, 5, 5]
        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry) #shape of spt : [75, 1, 5, 5]
        way = spt.shape[0]
        num_qry = qry.shape[0]

#----------------------------------cat--------------------------------------#
        channels = [self.channels[i] for i in self.hyperpixel_ids]
        spt_feats = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1).view(-1,*spt.size()[1:]) #shape of spt_feats [75x25,1,5,5]
        qry_feats = qry.unsqueeze(1).repeat(1, way, 1, 1, 1).view(-1,*qry.size()[1:]) #[75x25,1,5,5]
        spt_feats = torch.split(spt_feats,channels,dim=1)
        qry_feats = torch.split(qry_feats,channels,dim=1)
        corrs = []
        spt_feats_proj = []
        qry_feats_proj = []
        for i, (src, tgt) in enumerate(zip(spt_feats, qry_feats)):
            # corr = self.get_correlation_map(src, tgt, i, num_qry, way)#the shape of corr : [75x25, 25, 25]
            corr = self.corr(self.l2norm(src), self.l2norm(tgt)) #the shape of corr : [75x25, 25, 25]

            corrs.append(corr)
            spt_feats_proj.append(self.proj[i](src.flatten(2).transpose(-1, -2))) #[75x25,25,1]
            qry_feats_proj.append(self.proj[i](tgt.flatten(2).transpose(-1, -2))) #[75x25,25,1]

        spt_feats = torch.stack(spt_feats_proj, dim=1) #[75x25,1,25,1]
        qry_feats = torch.stack(qry_feats_proj, dim=1) #[75x25,1,25,1]
        corr = torch.stack(corrs, dim=1) #[75x25,1,25,25]
        # corr = self.mutual_nn_filter(corr)
        refined_corr = self.decoder(corr, spt_feats, qry_feats).view(num_qry,way,*[self.feature_size]*4)
        corr_s = refined_corr.view(num_qry, way, self.feature_size*self.feature_size, self.feature_size,self.feature_size)
        corr_q = refined_corr.view(num_qry, way, self.feature_size,self.feature_size, self.feature_size*self.feature_size)
        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr_s = self.gaussian_normalize(corr_s, dim=2)
        corr_q = self.gaussian_normalize(corr_q, dim=4)

        # applying softmax for each side
        corr_s = F.softmax(corr_s / self.args.temperature_attn, dim=2)
        corr_s = corr_s.view(num_qry, way, self.feature_size,self.feature_size, self.feature_size,self.feature_size)
        corr_q = F.softmax(corr_q / self.args.temperature_attn, dim=4)
        corr_q = corr_q.view(num_qry, way, self.feature_size,self.feature_size, self.feature_size,self.feature_size)

        # suming up matching scores
        attn_s = corr_s.sum(dim=[4, 5])
        attn_q = corr_q.sum(dim=[2, 3])

        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

#----------------------------------cat--------------------------------------#


        # spt_attended = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        # qry_attended = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

#----------------------------------replace--------------------------------------#


        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])
        # self.visualize(spt_attended_pooled, qry_attended_pooled)

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        # similarity_matrix = -F.pairwise_distance(spt_attended_pooled.view(num_qry*self.args.way,-1), qry_attended_pooled.view(num_qry*self.args.way,-1), p=2).view(num_qry,self.args.way)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def visualize(self, t1, t2):

        
        t1 = t1.view(75 * 5, 1024).cpu().detach()
        t = t1[0].unsqueeze(0)
        offset = 1
        for i in range(5, 75 * 5, 5):
            t = torch.cat([t, t1[i + offset].unsqueeze(0)], dim=0)
            offset = (offset + 1) % 5
        label1 = [1,2,3,4,5]*15

        t2 = t2.view(75 * 5, 1024).cpu().detach()
        offset = 0
        for i in range(0,75*5,5):
            t = torch.cat([t, t2[i + offset].unsqueeze(0)], dim=0)
            offset = (offset + 1) % 5
        label2 = [6,7,8,9,10] * 15

        label = label1+label2
        
        
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
        feats = self.encoder(x)
        
        # the shape of x : [way*(shot+query),1,5,5]
        x = feats.reshape(feats.shape[0], 1, 5, 5)
        return x

    def plot_embedding(self, data, label, title):
        # plt.rc('font',family='Times New Roman')
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
        fig = plt.figure()		# 创建图形实例
        ax = plt.subplot(111)		# 创建子图
        # 遍历所有样本
        
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], '.' if label[i] <= 5 else 'X', color=plt.cm.Set1(label[i] % 5),
                    fontdict={'weight': 'bold', 'size': 28 if label[i] <= 5 else 14})
        plt.xticks(fontproperties="Times New Roman")		# 指定坐标的刻度
        plt.yticks(fontproperties="Times New Roman")
        # plt.xticks()		# 指定坐标的刻度
        # plt.yticks()
        plt.title("Distribution of task" + str(self.args.visualfile) + " generated by ML", fontsize=14)
        plt.savefig("/data/data-home/chenderong/work/MCNet/visualizes/" + str(self.args.visualfile) + ".pdf", format="pdf")
        # 返回值
        return fig
