import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet
from models.cca import CCA
from models.scr import SCR, SelfCorrelationComputation
from models.others.se import SqueezeExcitation
from models.others.lsa import LocalSelfAttention
from models.others.nlsa import NonLocalSelfAttention
from models.others.sce import SpatialContextEncoder
from torchvision import transforms as T
import numpy as np
from models.resnet import conv3x3

class AugmentedCNN(nn.Module):
    def __init__(self):
        super(AugmentedCNN, self).__init__()

        self.conv = conv3x3(640*4,640*4)
        self.bn = nn.BatchNorm2d(640*4)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out * residual

        return out





class Method(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim*4, self.args.num_class)
        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])
        self.augmented_cnn = AugmentedCNN()

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
        # creating augmented data by applying three kinds of transfrom
        hflipper = T.RandomHorizontalFlip(p=1) #水平翻转
        vflipper = T.RandomVerticalFlip(p=1) #上下翻转
        rotater = T.RandomRotation([-90,-90]) #顺时针90度翻转

        x_h = torch.stack([hflipper(x[i]) for i in range(len(x))])
        x_v = torch.stack([vflipper(x[i]) for i in range(len(x))])
        x_r = torch.stack([rotater(x[i]) for i in range(len(x))])


       

        x = self.encoder(x)
        x_h = self.encoder(x_h)
        x_v = self.encoder(x_v)
        x_r = self.encoder(x_r)

        if self.args.self_method:
            identity = x
            identity_h = x_h
            identity_v = x_v
            identity_r = x_r

            x = self.scr_module(x)
            x_h = self.scr_module(x_h)
            x_v = self.scr_module(x_v)
            x_r = self.scr_module(x_r)

            if self.args.self_method == 'scr':
                x = x + identity
                x_h = x_h + identity_h
                x_v = x_v + identity_v
                x_r = x_r + identity_r #[100,640,5,5]
            x = F.relu(x, inplace=True)
            x_h = F.relu(x_h, inplace=True)
            x_v = F.relu(x_v, inplace=True)
            x_r = F.relu(x_r, inplace=True)

            x = torch.cat([x,x_h,x_v,x_r],dim=1) #[100,640*4,5,5]

            #再过一下卷积层


            x = self.augmented_cnn(x)

            # import sys
            # sys.exit(0)


            return x
