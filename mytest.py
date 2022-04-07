import torch
from models.cats import CATs
import torch.nn as nn
import torch.nn.functional as F

target = torch.randn(100,3,84,84)
source = torch.randn(100,3,84,84)

model = CATs()
x = model(target,source)



x.size()