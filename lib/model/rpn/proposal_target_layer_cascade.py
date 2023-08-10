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
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS) # (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)   # (0.1, 0.1, 0.2, 0.2)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)   # (1.0, 1.0, 1.0, 1.0)

    def forward(self, all_rois, gt_boxes, num_boxes):

        # all_rois  = (B, 2000, 5)
        # gt_boxes  = (B, 20, 5)
        # num_boxes = (B,)

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_() # (B, 20, 5)
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4] # (B, 20, 5)

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1) # (B, 2020, 5)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) # cfg.TRAIN.BATCH_SIZE = 128
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)) # cfg.TRAIN.FG_FRACTION = 0.25
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image # 32

        # all_rois          = (B, 2020, 5)
        # gt_boxes          = (B, 20, 5)
        # fg_rois_per_image = 32
        # rois_per_image    = 128
        # self._num_classes = 21
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        
        # labels              = (B, 128)
        # rois                = (B, 128, 5)
        # bbox_targets        = (B, 128, 4)
        # bbox_inside_weights = (B, 128, 4)

        bbox_outside_weights = (bbox_inside_weights > 0).float() # (= bbox_inside_weights)

        # rois                 = (B, 128, 5)
        # labels               = (B, 128)
        # bbox_targets         = (B, 128, 4)
        # bbox_inside_weights  = (B, 128, 4)
        # bbox_outside_weights = (B, 128, 4)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """

        # bbox_target_data = (B, 128, 4)
        # labels_batch     = (B, 128)
        # num_classes      = 21

        batch_size = labels_batch.size(0) # B
        rois_per_image = labels_batch.size(1) # 128
        clss = labels_batch # (B, 128)
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_() # (B, 128, 4) zero-init
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_() # (B, 128, 4) zero-init

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0: # 전체가 bg인 경우 고려하지 않음
                continue
            # 해당 batch에서 fg인 샘플만 선택하여 bbox_inside_weights을 1로 설정
            # 즉, fg인 샘플에 대해서만 bbox_reg loss를 계산함
            inds = torch.nonzero(clss[b] > 0).view(-1) 
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS # (1.0, 1.0, 1.0, 1.0)
            
        # bbox_targets        = (B, 128, 4)
        # bbox_inside_weights = (B, 128, 4)

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """

        # all_rois          = (B, 2020, 5)
        # gt_boxes          = (B, 20, 5)
        # fg_rois_per_image = 32
        # rois_per_image    = 128
        # num_classes       = 21

        # overlaps: (rois x gt_boxes)

        # 전체 IoU 계산
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        # overlaps = (B, 2020, 20)

        # 각 roi에 대해 가장 IoU가 큰 gt_box 식별
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        # max_overlaps  = 최대 IoU 값 (B, 2020)
        # gt_assignment = 최대 IoU 값을 갖는 gt_box indices (B, 2020)

        batch_size = overlaps.size(0) # B
        num_proposal = overlaps.size(1) # 2020
        num_boxes_per_img = overlaps.size(2) # 20

        # offset 적용
        offset = torch.arange(0, batch_size)*gt_boxes.size(1) # [0, 1, ..., B] * 20 = [0, 20, ..., 20B]
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment # (B, 1) + (B, 2020) = (B, 2020)

        # changed indexing way for pytorch 1.0
        labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        # gt_boxes[:,:,4] = class label만 추출 (B, 20)
        # gt_boxes[:,:,4].contiguous().view(-1) (20B,)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()       # (B, 128)
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()   # (B, 128, 5)
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_() # (B, 128, 5)
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        # 만약 한 batch에서 fg,bg의 비율을 조정함
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1) # gt와의 IoU가 fg thresold 이상인 rois만 선택
            fg_num_rois = fg_inds.numel() # fg의 수

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1) # gt와의 IoU가 bg thresold 이상, fg thresold 미만인 rois만 선택
            bg_num_rois = bg_inds.numel() # bg의 수

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                # 최대 fg 수를 넘지 않도록 조절함
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois) 

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]] # (nfg,)

                # sampling bg
                # 변경된 fg 수에 따라 bg의 수도 변경
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image 

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num] # (nbg,)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # 전체 batch가 fg로만 구성되는 경우 처리
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num] # (nfg,)
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # 전체 batch가 bg로만 구성되는 경우 처리
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num] # (nbg,)
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0) # (128,) (nfg + nbg = rois_per_image = 128)

            # Select sampled values from various arrays:
            # 처리 결과를 반영하여 한 배치에서 fg/bg 샘플링
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            # bg의 경우 class label을 0으로 지정
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0 

            # 마찬가지로 rois를 샘플링
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            # 마찬가지로 gt_boxes를 샘플링
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        # bbox reg를 위한 label 생성
        # rois_batch[:,:,1:5]    = (B, 128, 4)
        # gt_rois_batch[:,:,1:5] = (B, 128, 4)
        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])
        # bbox_target_data = (B, 128, 4)

        # loss weight을 고려
        # bbox_target_data = (B, 128, 4)
        # labels_batch     = (B, 128,)
        # num_classes      = 21
        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        # labels_batch        = (B, 128,)
        # rois_batch          = (B, 128, 5)
        # bbox_targets        = (B, 128, 4)
        # bbox_inside_weights = (B, 128, 4)
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
