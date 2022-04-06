import torch
import torchvision.models as models
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.backbone = models.resnet18(pretrained=True)
    def forward(self,img):
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        for lid in range(1,5):
            for bid in range(2):
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


        return feat

