from models.dataloader.mini_imagenet import MiniImageNet
from common.utils import parse_args
from matplotlib import pyplot as plt
from torchvision import transforms as T
import torch

args = parse_args("val")
data = MiniImageNet("val",args)
print(data[0][0].size())
rotater = T.RandomRotation([-90,-90])
data_r = torch.Tensor([rotater(data[i][0]) for i in range(len(data))])
# print(p.size())
# p.save("hahah.jpg")
# x.save("yeye.jpg")

