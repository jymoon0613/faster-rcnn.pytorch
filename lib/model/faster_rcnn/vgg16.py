# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # ! VGG-16 backbone은 마지막 max_pool 전까지만 사용함
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # ! Fast R-CNN detection head 정의
    # ! 우선 RPN의 출력 features를 4096차원으로 매핑
    self.RCNN_top = vgg.classifier # ! 25088 -> 4096

    # ! Classification head는 feature vectors로부터 class prediction 수행
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes) # ! 4096 -> 21

    # ! Regression head는 feature vectors로부터 bbox regression 수행
    if self.class_agnostic: # ! True
      self.RCNN_bbox_pred = nn.Linear(4096, 4) # ! 4096 -> 4
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)     

  def _head_to_tail(self, pool5):
    
    # ! pool5 = (128B, 512, 7, 7)

    pool5_flat = pool5.view(pool5.size(0), -1) # ! (128B, 25088)
    fc7 = self.RCNN_top(pool5_flat) # ! (128B, 4096)

    # ! fc7 = (128B, 4096)

    return fc7

