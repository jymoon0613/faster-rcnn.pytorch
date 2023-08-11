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
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS) # ! (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)   # ! (0.1, 0.1, 0.2, 0.2)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)   # ! (1.0, 1.0, 1.0, 1.0)

    def forward(self, all_rois, gt_boxes, num_boxes):
        
        # ! detection loss 계산을 위한 target 생성
        # ! all_rois  = (B, 2000, 5) -> 2000개의 region proposals가 (batch_num, x1, y1, x2, y2)의 형태로 저장되어있음
        # ! gt_boxes  = (B, 20, 5)   -> ground-truth bboxes
        # ! num_boxes = (B,)         -> number of ground-truth bboxes

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # ! 결과 저장을 위한 tensor 정의
        # ! 0으로 초기화
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_() # ! (B, 20, 5)

        # ! gt_boxes의 좌표 저장
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4] # ! (B, 20, 5)

        # Include ground-truth boxes in the set of candidate rois
        # ! gt_boxes와 모든 region proposals를 concat함
        all_rois = torch.cat([all_rois, gt_boxes_append], 1) # ! (B, 2020, 5)

        # ! 파라미터 설정
        # ! cfg.TRAIN.BATCH_SIZE = 128
        # ! cfg.TRAIN.FG_FRACTION = 0.25
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)  # ! 128
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)) # ! 32
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image # ! 32

        # ! training targets 생성
        # ! all_rois          = (B, 2020, 5)
        # ! gt_boxes          = (B, 20, 5)
        # ! fg_rois_per_image = 32
        # ! rois_per_image    = 128
        # ! self._num_classes = 21
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        
        # ! labels              = (B, 128,)   -> 각 rois에 대응되는 gt_cls_targets
        # ! rois                = (B, 128, 5) -> 최종적으로 선택된 rois
        # ! bbox_targets        = (B, 128, 4) -> bbox_reg loss gt_bbox_reg_targets
        # ! bbox_inside_weights = (B, 128, 4) -> bbox_reg loss 가중치

        bbox_outside_weights = (bbox_inside_weights > 0).float() # ! = bbox_inside_weights과 동일하게 설정

        # ! rois                 = (B, 128, 5) -> 최종적으로 선택된 rois
        # ! labels               = (B, 128)    -> 각 rois에 대응되는 gt_cls_targets
        # ! bbox_targets         = (B, 128, 4) -> bbox_reg loss gt_bbox_reg_targets
        # ! bbox_inside_weights  = (B, 128, 4) -> bbox_reg loss 가중치
        # ! bbox_outside_weights = (B, 128, 4) -> bbox_reg loss 가중치 (inside와 동일)

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

        # ! bbox_target_data = (B, 128, 4)
        # ! labels_batch     = (B, 128)
        # ! num_classes      = 21

        batch_size = labels_batch.size(0) # ! B
        rois_per_image = labels_batch.size(1) # ! 128
        clss = labels_batch # ! (B, 128)

        # ! 결과 저장을 위한 tensor 정의
        # ! bbox_targets: fg sample인 경우 기존의 (tx, ty, tw, th)를 유지하며, bg sample인 경우 0으로 재할당함
        # ! bbox_inside_weights: fg sample인 경우 1을 할당하여 bbox_reg loss 계산에서 사용하며, bg sample인 경우 0을 할당하여 bbox_reg 계산 안함
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_() # ! (B, 128, 4) zero-init
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_() # ! (B, 128, 4) zero-init

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            # ! batch가 bg로만 구성된 경우는 고려하지 않음
            if clss[b].sum() == 0: 
                continue

            # ! 해당 batch에서 fg인 샘플만 선택하여 bbox_inside_weights을 1로 설정
            # ! 즉, fg인 샘플에 대해서만 bbox_reg loss를 계산함
            inds = torch.nonzero(clss[b] > 0).view(-1) 
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                # ! 이때 (tx*, ty*, tw*, th*) 모두 weights=1로 동일한 크기로 고려함
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS # ! (1.0, 1.0, 1.0, 1.0)
            
        # ! bbox_targets        = (B, 128, 4)
        # ! bbox_inside_weights = (B, 128, 4)

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

        # ! all_rois          = (B, 2020, 5)
        # ! gt_boxes          = (B, 20, 5)
        # ! fg_rois_per_image = 32
        # ! rois_per_image    = 128
        # ! num_classes       = 21

        # overlaps: (rois x gt_boxes)

        # ! all_rois(region proposals + gt_boxes)와 gt_boxes 간의 IoU 계산
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        # ! overlaps = (B, 2020, 20)
        # ! 모든 2020개의 rois와 gt_boxes 간의 IoU 값이 저장되어 있음

        # ! 모든 rois에 대해 각 roi와 가장 IoU가 큰 gt_box 식별
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        # ! max_overlaps  = 최대 IoU 값 (B, 2020)
        # ! gt_assignment = 최대 IoU 값을 갖는 gt_box indices (B, 2020)

        batch_size = overlaps.size(0) # ! B
        num_proposal = overlaps.size(1) # ! 2020
        num_boxes_per_img = overlaps.size(2) # ! 20

        # ! gt_assignment에 offset 적용
        offset = torch.arange(0, batch_size)*gt_boxes.size(1) # ! [0, 1, ..., B-1] * 20 = [0, 20, ..., 20(B-1)]
        # ! offset: [0, 20, ..., 20(B-1)]
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment # ! (B, 1) + (B, 2020) = (B, 2020)
        # ! gt_assignment: B개의 batch, 2020개 rois는 각각 0 ~ 19 사이의 index를 가짐
        # ! offset + gt_assignment: B개의 batch, 2020개 rois는 각각 0 ~ 20(B-1)+19 사이의 index를 가짐
        # ! 만약 batch_size = 4라면 offset + gt_assignment에서 B개의 batch, 2020개 rois는 각각 0 ~ 79 사이의 index를 가짐 

        # changed indexing way for pytorch 1.0
        # ! cls_label 정의
        # ! 전체 rois에 대해 IoU가 최대가 되는 gt_box의 cls_labe을 할당
        labels = gt_boxes[:,:,4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1) # ! (B, 2020)

        # ! 결과 저장을 위한 tensor 정의
        # ! zero-initialization
        # ! labels_batch:    조건을 충족하는 rois의 gt_cls_label 정보 저장
        # ! rois_batch:      조건을 충족하는 rois 정보 저장
        # ! gt_rois_batch:   조건을 충족하는 rois에 대응하는 gt_boxes 정보 저장
        labels_batch = labels.new(batch_size, rois_per_image).zero_()       # ! (B, 128)
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()   # ! (B, 128, 5)
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_() # ! (B, 128, 5)
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        # ! 한 batch에서의 fg,bg의 비율을 조정함
        for i in range(batch_size):
            
            # ! gt와의 IoU가 fg thresold 이상인 rois만 선택
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1) 
            fg_num_rois = fg_inds.numel() # ! fg의 수

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            # ! gt와의 IoU가 bg thresold 이상, fg thresold 미만인 rois만 선택
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1) 
            bg_num_rois = bg_inds.numel() # ! bg의 수

            # ! batch 내에 positive/negative samples 모두 존재하는 경우
            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                # ! 최대 fg 수를 넘지 않도록 조절함
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois) 
                # ! 목표 fg 수 fg_rois_per_this_image = nfg

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]] # ! (nfg,)

                # sampling bg
                # ! 변경된 fg 수에 따라 bg의 수도 변경
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image # ! 128 - nfg
                # ! 목표 bg 수 bg_rois_per_this_image = nbg

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num] # ! (nbg,)

                # ! nfg + nbg = rois_per_image = 128

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # ! 전체 batch가 fg로만 구성되는 경우 처리
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num] # ! (nfg,) = (rois_per_image,) = (128,)
                fg_rois_per_this_image = rois_per_image # ! 128
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # ! 전체 batch가 bg로만 구성되는 경우 처리
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num] # ! (nbg,) = (rois_per_image,) = (128,)
                bg_rois_per_this_image = rois_per_image # ! 128
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            # ! 처리 결과를 바탕으로 샘플링할 fg/bg indices를 정의함
            keep_inds = torch.cat([fg_inds, bg_inds], 0) # ! (128,)

            # Select sampled values from various arrays:
            # ! 결과에 따라 한 batch에서 cls_label 샘플링
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            # ! bg의 경우 class label을 0(__background__)으로 지정
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0 

            # ! 마찬가지로 결과에 따라 rois를 샘플링
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            # ! 마찬가지로 결과에 따라 각 rois에 대응되는 gt_boxes를 샘플링
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]] # (128,)

        # ! bbox reg를 위한 label 생성
        # ! rois_batch[:,:,1:5]    = (B, 128, 4) -> 샘플링된 rois 좌표
        # ! gt_rois_batch[:,:,1:5] = (B, 128, 4) -> 샘플링된 rois에 대응되는 gt_boxes 좌표
        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])
        # ! bbox_target_data = (B, 128, 4) -> 각 batch 내에 존재하는 128개의 rois에 대해 각 roi와 매칭된 gt_box와 계산된 gt_bbox targets

        # ! loss weight을 계산
        # ! bbox_target_data = (B, 128, 4)
        # ! labels_batch     = (B, 128)
        # ! num_classes      = 21
        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        # ! bbox_targets: fg sample의 경우 기존의 값을 유지, bg sample은 0으로 마스킹된 bbox_reg target
        # ! bbox_inside_weights: fg sample의 경우 (1, 1, 1, 1)로 bbox_reg loss 계산에 사용되며, bg sample의 경우 (0, 0, 0, 0)으로 bbox_reg loss에서 고려 X

        # ! labels_batch        = (B, 128,)   -> 각 rois에 대응되는 gt_cls_targets
        # ! rois_batch          = (B, 128, 5) -> 최종적으로 선택된 rois
        # ! bbox_targets        = (B, 128, 4) -> bbox_reg loss gt_bbox_reg_targets
        # ! bbox_inside_weights = (B, 128, 4) -> bbox_reg loss 가중치

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
