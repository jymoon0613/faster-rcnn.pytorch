from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # ! 512
        self.anchor_scales = cfg.ANCHOR_SCALES # ! [8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS # ! [0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0] # ! 16

        # define the convrelu layers processing input feature map
        # ! 초기 convolution을 위한 layer 정의
        # ! self.RPN_Conv = nn.Conv2d(512, 512, 3, 1, 1, bias=True)
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # ! RPN class prediction (fg/bg) 수행을 위한 layer 정의
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # ! 2 (bg/fg) * 9 (anchors) = 18
        # ! self.RPN_cls_score = nn.Conv2d(512, 18, 1, 1, 0)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # ! RPN bbox regression (tx, ty, tw, th) 수행을 위한 layer 정의
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # ! 4 (coords) * 9 (anchors) = 36
        # ! self.RPN_bbox_pred = nn.Conv2d(512, 36, 1, 1, 0)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        # ! RPN의 cls, bbox 예측값을 바탕으로 region proposals를 생성하는 layer 정의
        # ! self.RPN_proposal = _ProposalLayer(16, [8,16,32], [0.5,1,2])
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        # ! RPN의 training target을 생성하는 layer 정의
        # ! self.RPN_anchor_target = _AnchorTargetLayer(16, [8,16,32], [0.5,1,2])
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        
        # ! Feature maps를 입력받아 region proposals를 생성함
        # ! base_feat = (B, 512, 37, 37)
        # ! im_info   = (B, 3)
        # ! gt_boxes  = (B, 20, 5)
        # ! num_boxes = (B,)

        batch_size = base_feat.size(0) # ! B

        # return feature map after convrelu layer
        # base_feat = (B, 512, 37, 37)
        # ! 초기 convolution 적용 (+ ReLU)
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # ! rpn_conv1 = (B, 512, 37, 37)

        # get rpn classification score
        # ! RPN class prediction 수행
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)
        # ! rpn_cls_score = (B, 18, 37, 37)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # ! rpn_cls_score_reshape = (B, 2, 333, 37)

        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1) # ! bg/fg에 대한 softmax
        # ! rpn_cls_prob_reshape = (B, 2, 333, 37)

        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        # ! rpn_cls_prob = (B, 18, 37, 37)

        # get rpn offsets to the anchor boxes
        # ! RPN bbox regression 수행
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
        # ! rpn_bbox_pred = (B, 36, 37, 37)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST' # ! training을 가정

        # ! 예측값을 바탕으로 region proposals를 생성
        # ! rpn_cls_prob  = (B, 18, 37, 37) -> 모든 feature map positions, 모든 anchor boxes에 대한 objectness score 예측값
        # ! rpn_bbox_pred = (B, 36, 37, 37) -> 모든 feature map positions, 모든 anchor boxes에 대한 bbox offset 예측값
        # ! im_info       = (B, 3)
        # ! cfg_key       = 'TRAIN'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))
        
        # ! rois = (B, 2000, 5) -> 2000개의 proposals에 대해 (batch_num, x1, y1, x2, y2)로 구성됨

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # training labels를 생성
            # rpn_cls_score = (B, 18, 37, 37) 모든 feature map positions, 모든 anchor boxes에 대한 objectness score 예측값
            # gt_boxes      = (B, 20, 5)      ground-truth bboxes의 좌표
            # im_info       = (B, 3)          image resolution 정보     
            # num_boxes     = (B,)            ground-truth bboxes의 수
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            # rpn_data[0] = labels               (B, 1, 333, 37)
            # rpn_data[1] = bbox_targets         (B, 36, 37, 37)
            # rpn_data[2] = bbox_inside_weights  (B, 36, 37, 37)
            # rpn_data[3] = bbox_outside_weights (B, 36, 37, 37)

            # Objectness score loss 계산
            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # (B, 12321, 2)
            rpn_label = rpn_data[0].view(batch_size, -1) # (B, 12321)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1)) # (Bn,)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep) # (Bn, 2)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data) # (Bn,)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label) # -> scalar loss value
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # bbox reg loss 계산
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss  
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)   # (B, 36, 37, 37)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights) # (B, 36, 37, 37)
            rpn_bbox_targets = Variable(rpn_bbox_targets)                 # (B, 36, 37, 37)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3]) # -> scalar loss value

        return rois, self.rpn_loss_cls, self.rpn_loss_box
