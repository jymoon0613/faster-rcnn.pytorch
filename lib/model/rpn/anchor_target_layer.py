from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride # ! 16
        self._scales = scales # ! [8,16,32]
        anchor_scales = scales # ! [0.5,1,2]

        # ! anchor box 생성
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        # ! self._anchors = (9, 4) -> 서로 다른 aspect ratio, 서로 다른 scale의 anchor boxes에 대한 좌표 (x1, y1, x2, y2)

        self._num_anchors = self._anchors.size(0) # ! 9

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        # ! input[0] = rpn_cls_score = (B, 18, 37, 37) -> 모든 feature map positions, 모든 anchor boxes에 대한 objectness score 예측값
        # ! input[1] = gt_boxes      = (B, 20, 5)      -> ground-truth bboxes의 좌표
        # ! input[2] = im_info       = (B, 3)          -> image resolution 정보     
        # ! input[3] = num_boxes     = (B,)            -> ground-truth bboxes의 수

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3) # ! 37, 37

        batch_size = gt_boxes.size(0) # ! B

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3) # ! 37, 37

        # ! feature map의 각 좌표를 원본 이미지 상의 좌표로 변환
        shift_x = np.arange(0, feat_width) * self._feat_stride # ! [0, 1, 2, ..., 37] * 16 -> [0, 16, 32, ..., 576] (37,)
        shift_y = np.arange(0, feat_height) * self._feat_stride # ! [0, 1, 2, ..., 37] * 16 -> [0, 16, 32, ..., 576] (37,)

        # ! 그리드 좌표로 변환
        shift_x, shift_y = np.meshgrid(shift_x, shift_y) # ! (37,37), (37,37)

        # ! bbox 형식으로 변환 -> [[x1, y1, x2, y2], ...]
        # ! 이때 x1=x2, y1=y2임 (특정 position을 의미하므로)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()
        # ! shifts = (1369, 4)

        A = self._num_anchors # ! 9
        K = shifts.size(0) # ! 37 * 37 = 1369

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.

        # ! 1369(37x37)의 모든 reference positions 상에서 9개의 anchor boxes를 정의함
        # ! 즉, anchors는 모든 reference positions 상에서 9개의 anchor boxes 좌표를 담고 있음 (1369, 9, 4)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # ! self._anchors = (9, 4)
        # ! self._anchors.view(1, A, 4) = (1, 9, 4)
        # ! shifts = (1369, 4)
        # ! shifts.view(K, 1, 4) = (1369, 1, 4)
        # ! broadcasting: anchors = (1, 9, 4) + (1369, 1, 4) = (1369, 9, 4)

        all_anchors = all_anchors.view(K * A, 4) # ! reshape (12321, 4)

        total_anchors = int(K * A) # ! 1369 * 9 = 12321

        # ! 이미지 경계를 벗어나지 않는 anchors만 선택함 (n개라고 가정)
        # ! self._allowed_border = 0
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        # ! 결과에 따라 남길 anchors만 선택
        inds_inside = torch.nonzero(keep).view(-1) # ! (n,)
        anchors = all_anchors[inds_inside, :] # ! (n, 4)

        # ! objectness scores loss 계산을 위한 labels 정의 (B, n) (-1로 채움)
        # ! label = 1 -> positive sample, label =  0 -> negative sample, label = -1 -> 고려하지 않음
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)

        # ! bbox_inside_weights 정의 (B, n) (0으로 채움)
        # ! bbox_reg loss는 positive samples에 대해서만 정의되며 이를 반영하기 위한 weights임
        # ! 즉, weight = 1이면 positive sample로 bbox_reg_loss 계산을 수행하고, 0이면 negative sample로서 수행하지 않음
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        # ! bbox_outside_weights 정의 (B, n) (0으로 채움)
        # ! 마찬가지로 bbox_outsize_weights은 neagative samples에 대한 bbox_reg_loss를 결정
        # ! 본 예시에서는 negative samples에 대한 bbox_reg loss를 사용하지 않음
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        # ! 먼저, n개의 anchor boxes와 모든 gt_boxes 간의 IoU를 계산
        # ! 즉, anchor boxes마다 대응하는 gt_box를 하나씩 할당함
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        # ! overlaps = (B, n, 20) -> 모든 n개의 anchor boxes에 대해 모든 20개의 gt_boxes 간의 IoU

        # ! n개의 anchor boxes에 대해 각 anchor box와 가장 IoU가 높은 gt_box 식별
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # ! max_overlaps = (B, n) -> IoU 값
        # ! argmax_overlaps = (B, n) -> gt_box indices

        # ! 20개의 gt_boxes에 대해 각 gt_box와 가장 IoU가 높은 anchor_box 식별
        gt_max_overlaps, _ = torch.max(overlaps, 1)
         # ! gt_max_overlaps = (B, 20) -> IoU 값

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES: # ! cfg.TRAIN.RPN_CLOBBER_POSITIVES = False
            # ! 각 anchor box 별로 할당된 gt_box에 대해 IoU가 IoU_bg 미만인 경우 negative sample로 사용 (label = 0)
            # ! cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 

        # ! 이전에 max 연산을 통해 anchor box 별로 IoU가 가장 큰 하나의 gt_box를 할당하였음
        # ! 하지만, 하나의 anchor box에 대해 IoU가 max인 gt_box가 여러 개일 수도 있음 
        # ! 이를 식별하기 위해 각 anchor box에 대해 IoU가 가장 크면서 같은 anchor boxes를 모두 식별
        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # ! n 개의 anchor box 각각에 대해 할당된 gt_boxes의 개수 계산
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2) # ! (B, n)

        # ! 
        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        # ! 각 anchor box 별로 할당된 gt_box에 대해, IoU가 IoU_fg 이상인 경우 positive sample로 사용 (label = 1)
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES: # ! cfg.TRAIN.RPN_CLOBBER_POSITIVES = False
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # ! 한 batch 내에서 positive/negative samples의 비율을 조정
        # ! cfg.RPN_FG_FRACTION = 0.5
        # ! cfg.TRAIN.RPN_BATCHSIZE = 256
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE) # ! 0.5 * 256 = 128

        sum_fg = torch.sum((labels == 1).int(), 1) # ! fg의 개수 (B,)
        sum_bg = torch.sum((labels == 0).int(), 1) # ! bg의 개수 (B,)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            # ! 만약 한 batch에서 fg가 비율에 맞지 않게 많다면 fg를 일부만 샘플링함
            if sum_fg[i] > num_fg:
                # ! fg의 indices를 식별
                fg_inds = torch.nonzero(labels[i] == 1).view(-1) 

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                
                # ! 랜덤 샘플링 후, 샘플링된 fg를 제외한 나머지 fg는 고려하지 않음 (label = -1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            # ! bg의 개수 식별 = 전체 batch size - fg의 개수
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            # ! 마찬가지로 bg에 대해서도 진행
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        # ! batch size offset 지정
        offset = torch.arange(0, batch_size)*gt_boxes.size(1) # ! [0, 1, ..., B] * 20 = [0, 20, ..., 20B]

        # ! bbox reg 예측 타겟 (tx*, ty*, tw*, th*) 생성
        # ! offset 적용 (B, n)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps) 
        
        # ! anchors = (n, 4)
        # ! gt_boxes.view(-1,5) = (20B, 5)
        # ! gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :] = (Bn, 5)
        # ! gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5) = (B, n, 5)
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        # ! bbox_targets = (B, n, 4)

        # use a single value instead of 4 values for easy index.
        # positive smaple에 대한 가중치 설정
        # cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0] # (B, n)

        # cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights # (B, n)
        bbox_outside_weights[labels == 0] = negative_weights # (B, n)
        
        # total_anchors        = 12321
        # ind_insize           = (n,)
        # labels               = (B, n)
        # bbox_targets         = (B, n, 4)
        # bbox_inside_weights  = (B, n)
        # bbox_outside_weights = (B, n)
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # labels               = (B, 12321)     -> 이때 선택된 n개의 anchors에 대해서만 labels 부여하고 나머지는 -1로 할당하여 고려하지 않음
        # bbox_targets         = (B, 12321, 4)  -> 이때 선택된 n개의 anchors에 대해서만 bbox_targets를 사용하고 부여하고 나머지는 0으로 할당하여 고려하지 않음
        # bbox_inside_weights  = (B, 12321)     -> 이때 선택된 n개의 anchors에 대해서만 bbox_targets를 사용하고 부여하고 나머지는 0으로 할당하여 고려하지 않음
        # bbox_outside_weights = (B, 12321)     -> 이때 선택된 n개의 anchors에 대해서만 bbox_targets를 사용하고 부여하고 나머지는 0으로 할당하여 고려하지 않음
        # 이때 선택된 n개의 anchors에 대해서만 labels 부여하고 나머지는 -1로 할당하여 고려하지 않음

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous() # (B, 9, 37, 37)
        labels = labels.view(batch_size, 1, A * height, width) # (B, 1, 333, 37)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous() # (B, 36, 37, 37)
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1) # 12321
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4) # (B, 12321, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous() # (B, 36, 37, 37)

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4) # (B, 12321, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous() # (B, 36, 37, 37)
        outputs.append(bbox_outside_weights)

        # output[0] = labels               (B, 1, 333, 37)
        # output[1] = bbox_targets         (B, 36, 37, 37)
        # output[2] = bbox_inside_weights  (B, 36, 37, 37)
        # output[3] = bbox_outside_weights (B, 36, 37, 37)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
