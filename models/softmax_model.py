
import torch
from torch import nn
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from models.Asoftmax_linear import AngleLinear
import pdb

class SoftmaxModule(nn.Module):
    def __init__(self, config):
        super(SoftmaxModule, self).__init__()
        self.num_classes = config.numcls
        self.trans_layer = nn.Sequential(
                                         nn.Linear(50, 32), 
                                         nn.ReLU(),
                                         nn.Linear(32, 4), 
                                         nn.ReLU(),
                                       )
        self.trans_layer_2 = nn.Linear(4*2048, 2048)
        self.Acls = AngleLinear(config.bank_dim, self.num_classes) 


    def forward(self, feat):
        bs = feat.size(0)
        mem_down_feat = self.trans_layer(feat)
        mem_cls_feat = mem_down_feat.view(bs, -1)
        mem_cls_feat = self.trans_layer_2(mem_cls_feat)
        out = self.Acls(mem_cls_feat)
        return out
 







