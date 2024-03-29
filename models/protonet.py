import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet12 import ResNet12
from models.cca import CCA
from models.scr import SCR, SelfCorrelationComputation
from models.others.se import SqueezeExcitation
from models.others.lsa import LocalSelfAttention
from models.others.nlsa import NonLocalSelfAttention
from models.others.sce import SpatialContextEncoder
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import sys


class ProtoNet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet12(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        if self.args.self_method == 'scr':
            corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            self_block = SCR(planes=planes, stride=stride)
        elif self.args.self_method == 'sce':
            planes = [640, 64, 64, 640]
            self_block = SpatialContextEncoder(planes=planes, kernel_size=kernel_size[0])
        elif self.args.self_method == 'se':
            self_block = SqueezeExcitation(channel=planes[0])
        elif self.args.self_method == 'lsa':
            self_block = LocalSelfAttention(in_channels=planes[0], out_channels=planes[0], kernel_size=kernel_size[0])
        elif self.args.self_method == 'nlsa':
            self_block = NonLocalSelfAttention(planes[0], sub_sample=False)
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
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.smlrty(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def smlrty(self, spt, qry):

        num_qry, way, shot = self.args.query, self.args.way, self.args.shot
        spt = spt.squeeze(0)
        

        # shifting channel activations by the channel mean
        spt = self.gaussian_normalize(spt,dim=1)
        qry = self.gaussian_normalize(qry,dim=1)
        spt_ = spt.mean(dim=[-1,-2]).contiguous().view(shot,way,-1).mean(dim=0)
        qry_ = qry.mean(dim=[-1,-2])
        spt_ext = spt_.unsqueeze(0).repeat(num_qry*way,1,1).contiguous().view(num_qry*way*way,-1)
        qry_ext = qry_.unsqueeze(1).repeat(1,way,1).contiguous().view(num_qry*way*way,-1)
        # self.visualize(spt_ext,qry_ext)

        similarity_matrix = -F.pairwise_distance(spt_ext,qry_ext,p=2).contiguous().view(num_qry*way,way)


        if self.training:
            return similarity_matrix , self.fc(qry_)
        else:
            return similarity_matrix

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x


    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x
            x = self.scr_module(x)

            if self.args.self_method == 'scr':
                x = x + identity
            x = F.relu(x, inplace=True)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x

    def visualize(self, t1, t2):
        t1 = t1.view(75*5, 1024).cpu().detach()
        label1 = [1,2,3,4,5] * 75
        t2 = t2.view(75 * 5, 1024)[1:26:5,:].cpu().detach()
        label2 = [7] * 5
        t = torch.cat([t1,t2],dim=0)
        label = label1 + label2
        ts = TSNE(n_components=2, init="pca", random_state=0)
        result = ts.fit_transform(t)
        self.plot_embedding(result, label, 't-SNE Embedding of digits')
        sys.exit()

    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
        fig = plt.figure()		# 创建图形实例
        ax = plt.subplot(111)		# 创建子图
        # 遍历所有样本
        dotdict = {1:'o',2:'x',3:'*',4:'.',5:'^',7:'@'}
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], dotdict[label[i]], color=plt.cm.Set1(label[i] / 10),
                    fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()		# 指定坐标的刻度
        plt.yticks()
        plt.title(title, fontsize=14)
        plt.savefig("/data/data-home/chenderong/work/MCNet/visualizes/a.jpg")
        # 返回值
        return fig
