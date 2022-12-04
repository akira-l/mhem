import os
import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
#import pretrainedmodels

from config import pretrained_model

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            raise Exception('no pretrainedmodels package now')
            #self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)

        if self.backbone_arch == 'resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            #self.model_1 = nn.Sequential(*list(self.model.children())[:-3])
            #self.model_2 = nn.Sequential(*list(self.model.children())[-3])

        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool_2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        self.relu = nn.ReLU()

    def forward(self, inputs, labels=None, img_names=None):

        cls_feat = self.model(inputs)
        bs, dim, fw, fh = cls_feat.size()

        feat_hm = F.normalize(self.relu(cls_feat).sum(1))
        feat_hm_view = feat_hm.view(bs, -1)
        feat_view = cls_feat.view(bs, dim, -1)
        pick_val, pick_pos = torch.topk(feat_hm_view, fw*fh//4)
        gather_feat = [sub_feat[:, sub_pos] for sub_feat, sub_pos in zip(feat_view, pick_pos)]
        gather_feat = torch.stack(gather_feat)
        avg_top_feat = self.avgpool_1d(gather_feat).view(bs, -1)

        avg_feat = self.avgpool_2d(cls_feat)
        fin_feat = avg_feat.view(bs, -1)
        out = self.classifier(fin_feat)

        return out, cls_feat, fin_feat, avg_top_feat #pick_val

