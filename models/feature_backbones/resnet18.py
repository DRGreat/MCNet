import torch
import torchvision.models as models
import torch.nn as nn
from operator import add
from functools import reduce, partial
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self,freeze=False, feature_size=16, hyperpixel_ids = [2,4,6,8]):
        super(ResNet18,self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        nbottlenecks = [2, 2, 2, 2]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids


    def forward(self,img):

        feats = []
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())


        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)

            if bid == 0 and self.backbone.__getattr__('layer%d' % lid)[bid].downsample != None:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)
        
        return feats

