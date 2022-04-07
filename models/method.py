import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet
from models.feature_backbones.resnet18 import ResNet18
from models.cca import CCA
from models.scr import SCR, SelfCorrelationComputation
from models.others.se import SqueezeExcitation
from models.others.lsa import LocalSelfAttention
from models.others.nlsa import NonLocalSelfAttention
from models.others.sce import SpatialContextEncoder


class Method(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        # self.encoder = ResNet(args=args)
        self.encoder = ResNet18(freeze=False)
        self.encoder_dim = 960
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.scr_module = nn.ModuleList(
            [
            self._make_scr_layer(planes=[64, 64, 64, 64, 64]),
            self._make_scr_layer(planes=[128, 64, 64, 64, 128]),
            self._make_scr_layer(planes=[256, 64, 64, 64, 256]),
            self._make_scr_layer(planes=[512, 64, 64, 64, 512])
            ]
        )
       

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
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        spt = spt.squeeze(0)

        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        way = spt.shape[0]
        num_qry = qry.shape[0]

        spt_attended = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry_attended = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

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

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)
        # similarity_matrix = -F.pairwise_distance(spt_attended_pooled.view(num_qry*self.args.way,-1), qry_attended_pooled.view(num_qry*self.args.way,-1), p=2).view(num_qry,self.args.way)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def get_4d_correlation_map(self, spt, qry):
        '''
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        '''
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        feats = self.encoder(x)
        
        for idx,x in enumerate(feats):
            if self.args.self_method:
                identity = x
                x = self.scr_module[idx](x)

                if self.args.self_method == 'scr':
                    x = x + identity
                x = F.relu(x, inplace=True)
            feats[idx] = x
        x = torch.cat(feats,dim=1)
        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x
