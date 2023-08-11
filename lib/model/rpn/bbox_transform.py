# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    # ! _AnchorTargetLayer
    # ! ex_rois = anchors = (n, 4) -> 이미지 boundary 내에 존재하는 n개의 anchor boxes
    # ! gt_rois = (B, n, 5) -> 각 batch 내에 존재하는 n개의 anchor boxes에 대해 각 anchor box와 매칭된 gt_box의 좌표
    # ! 현재 ex_rois.dim() == 2
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0          # ! anchor box의 너비 (n,)
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0         # ! anchor box의 높이 (n,)
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths               # ! anchor box의 중심좌표 x (n,)
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights              # ! anchor box의 중심좌표 y (n,)

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0    # ! gt box의 너비 (B, n)
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0   # ! gt box의 높이 (B, n)
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths            # ! gt box의 중심좌표 x (B, n)
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights           # ! gt box의 중심좌표 y (B, n)

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths    # ! tx* (B, n)
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights   # ! ty* (B, n)
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))    # ! tw* (B, n)
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights)) # ! th* (B, n)

    # ! _ProposalTargetLayer
    # ! ex_rois    = (B, 128, 4) -> 샘플링된 rois
    # ! gt_rois    = (B, 128, 4) -> 샘플링된 rois에 대응되는 gt_boxes
    # ! 현재 ex_rois.dim() == 3
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0  # ! rois의 너비 (B, 128)
        ex_heights = ex_rois[:,:, 3] - ex_rois[:, :, 1] + 1.0  # ! rois의 높이 (B, 128)
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths          # ! rois의 중심좌표 x (B, 128)
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights         # ! rois의 중심좌표 y (B, 128)

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0  # ! gt box의 너비 (B, 128)
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0 # ! gt box의 높이 (B, 128)
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths          # ! gt box의 중심좌표 x (B, 128)
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights         # ! gt box의 중심좌표 y (B, 128)

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths  # ! tx* (B, 128)
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights # ! ty* (B, 128)
        targets_dw = torch.log(gt_widths / ex_widths)   # ! tw* (B, 128)
        targets_dh = torch.log(gt_heights / ex_heights) # ! th* (B, 128)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)
    
    # ! _AnchorTargetLayer   -> targets = (B, n, 4)
    # ! _ProposalTargetLayer -> targets = (B, 128, 4)

    return targets

def bbox_transform_inv(boxes, deltas, batch_size):

    # ! RPN은 bbox regression을 위해 bbox의 중심좌표 (x, y)와 bbox의 너비/높이 (h, w)의 anchor box 기준 offset을 예측함 (tx, ty, tw, th)
    # ! tx = (x - x_a) / w_a
    # ! ty = (y - y_a) / h_a
    # ! tw = log(w / w_a)
    # ! th = log(h / h_a)
    # ! 이때 x_a, y_a, w_a, h_a는 anchor box의 중심좌표, 너비/높이임
    # ! 또한, ground-truth bbox reg labels (tx*, ty*, tw*, th*)는 다음과 같음
    # ! tx* = (x* - x_a) / w_a
    # ! ty* = (y* - y_a) / h_a
    # ! tw* = log(w* / w_a)
    # ! th* = log(h* / h_a)
    # ! 따라서, 예측된 bbox reg offset (tx, ty, tw, th)를 실제 proposals (x, y, w, h)로 변환해주는 과정이 요구됨

    # ! boxes      = (B, 12321, 4)
    # ! deltas     = (B, 12321, 4)
    # ! batch_size = B

    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0  # ! anchor boxes 너비 (B, 12321) - (B, 12321) = (B, 12321)
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0 # ! anchor boxes 높이 (B, 12321) - (B, 12321) = (B, 12321)
    ctr_x = boxes[:, :, 0] + 0.5 * widths  # ! anchor boxes의 중심 좌표 계산 x: (B, 12321) + 0.5 * (B, 12321) = (B, 12321)
    ctr_y = boxes[:, :, 1] + 0.5 * heights # ! anchor boxes의 중심 좌표 계산 y: (B, 12321) + 0.5 * (B, 12321) = (B, 12321)

    dx = deltas[:, :, 0::4] # ! tx (B, 12321, 1)
    dy = deltas[:, :, 1::4] # ! ty (B, 12321, 1)
    dw = deltas[:, :, 2::4] # ! tw (B, 12321, 1)
    dh = deltas[:, :, 3::4] # ! th (B, 12321, 1)

    # ! proposals (x, y, w, h)로 변환 (역변환, inverse)
    # ! x = tx * w_a + x_a
    # ! y = ty * h_a + y_a
    # ! w = exp(tw) * w_a
    # ! h = exp(th) * h_a
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)  # ! (B, 12321, 1) * (B, 12321, 1) + (B, 12321, 1) = (B, 12321, 1)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2) # ! (B, 12321, 1) * (B, 12321, 1) + (B, 12321, 1) = (B, 12321, 1)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)                # ! exp((B, 12321, 1)) * (B, 12321, 1) = (B, 12321, 1)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)               # ! exp((B, 12321, 1)) * (B, 12321, 1) = (B, 12321, 1)

    pred_boxes = deltas.clone() # ! (B, 12321, 4)
    
    # ! 계산된 proposals (x, y, w, h)를 바탕으로 다시 bbox 좌표 형태 (x1, y1, x2, y2)로 변환 
    # ! x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w 
    # ! y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # ! x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # ! y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    # ! pred_boxes = (B, 12321, 4)

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """

    # ! boxes        = (B, 12321, 4)
    # ! im_shape     = (B, 3)
    # ! batch_size   = B

    num_rois = boxes.size(1) # ! 12321

    # ! 좌표 값이 음수 -> 이미지 영역을 벗어남 -> 이미지 영역 내로 clip
    boxes[boxes < 0] = 0

    batch_x = im_shape[:, 1] - 1 # ! W - 1 = 600 - 1 = 599
    batch_y = im_shape[:, 0] - 1 # ! H - 1 = 600 - 1 = 599

    # ! x1, y1, x2, y3 모두에 대해 image boundary를 초과하는 경우 이미지 경계로 clip
    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    # ! boxes = (B, 12321, 4)

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_boxes.size(0) # ! B

    # ! _AnchorTargetLayer
    # ! anchors  = (n, 4)
    # ! gt_boxes = (B, 20, 5)
    # ! 현재 anchors.dim() == 2
    if anchors.dim() == 2:

        N = anchors.size(0) # ! n
        K = gt_boxes.size(1) # ! 20

        # ! anchors를 batch size만큼 확장
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous() # ! (B, n, 4)
        gt_boxes = gt_boxes[:,:,:4].contiguous() # ! bbox 좌표 값만 가져옴 (B, 20, 4)

        # ! gt_boxes의 크기 계산
        # ! 먼저 gt_boxes의 너비/높이 계산
        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1) # ! (B, 20) - (B, 20) + 1 = (B, 20)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1) # ! (B, 20) - (B, 20) + 1 = (B, 20)

        # ! 다음으로 gt_boxes의 크기 계산
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K) # ! (B, 20) * (B, 20) = (B, 20) -> (B, 1, 20)
        # ! gt_boxes_area = (B, 1, 20)

        # ! anchor boxes의 크기 계산
        # ! 먼저 anchor boxes의 너비/높이 계산
        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1) # ! (B, n) - (B, n) + 1 = (B, n)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1) # ! (B, n) - (B, n) + 1 = (B, n)

        # ! 다음으로 anchor boxes의 크기 계산
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1) # ! (B, n) * (B, n) = (B, n) -> (B, n, 1)
        # ! anchors_area = (B, n, 1)

        # ! 비정상적인 gt_boxes, anchor boxes 식별
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        # ! IoU 계산
        # ! 먼저, reshape 진행
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4) # ! (B, n, 1, 4) -> (B, n, 20, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4) # ! (B, 1, 20, 4) -> (B, n, 20, 4)

        # ! intersection의 너비/높이 계산
        # ! 이때 음수인 경우 필터링
        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1) # ! (B, n, 20)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1) # ! (B, n, 20)
        ih[ih < 0] = 0

        # ! 최종적으로 IoU 계산
        # ! 모든 n개의 anchor boxes와 모든 20개의 gt_boxes 간의 IoU
        ua = anchors_area + gt_boxes_area - (iw * ih) # ! (B, n, 1) + (B, 1, 20) - (B, n, 20) = (B, n, 20)
        overlaps = iw * ih / ua # ! (B, n, 20) * (B, n, 20) / (B, n, 20) = (B, n, 20)

        # mask the overlap here.
        # ! 이전에 식별한 비정상적인 gt_boxes, anchor boxes 처리
        # ! gt_area의 크기가 0인 경우 0으로, anchor_area가 0인 경우 -1로 처리
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

        # ! overlaps = (B, n, 20)

    # ! _ProposalTargetLayer
    # ! anchors  = (B, 2020, 5)
    # ! gt_boxes = (B, 20, 5)
    # ! 현재 anchors.dim() == 3
    elif anchors.dim() == 3:
        N = anchors.size(1) # ! 2020
        K = gt_boxes.size(1) #  ! 20

        # ! bbox 좌표만 추출
        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous() # ! (B, 2020, 4)

        gt_boxes = gt_boxes[:,:,:4].contiguous() # ! (B, 20, 4)

        # ! gt_boxes의 크기 계산
        # ! 먼저 gt_boxes의 너비/높이 계산
        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1) # ! (B, 20) - (B, 20) + 1 = (B, 20)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1) # ! (B, 20) - (B, 20) + 1 = (B, 20)

        # ! 다음으로 gt_boxes의 크기 계산
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K) # ! (B, 20) * (B, 20) = (B, 20) -> (B, 1, 20)
        # ! gt_boxes_area = (B, 1, 20)

        # ! roi boxes의 크기 계산
        # ! 먼저 roi boxes의 너비/높이 계산
        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1) # ! (B, 2020) - (B, 2020) + 1 = (B, 2020)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1) # ! (B, 2020) - (B, 2020) + 1 = (B, 2020)

        # ! 다음으로 roi boxes의 크기 계산
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1) # ! (B, 2020) * (B, 2020) = (B, 2020) -> (B, 2020, 1)
        # ! anchors_area = (B, 2020, 1)

        # ! 비정상적인 gt_boxes, roi boxes 식별
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        # ! IoU 계산
        # ! 먼저, reshape 진행
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4) # ! (B, 2020, 1, 4) -> (B, 2020, 20, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4) # ! (B, 1, 20, 4) -> (B, 2020, 20, 4)

        # ! intersection의 너비/높이 계산
        # ! 이때 음수인 경우 필터링
        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1) # ! (B, 2020, 20)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1) # ! (B, 2020, 20)
        ih[ih < 0] = 0

        # ! 최종적으로 IoU 계산
        # ! 모든 2020개의 rois와 모든 20개의 gt_boxes 간의 IoU
        ua = anchors_area + gt_boxes_area - (iw * ih) # ! (B, 2020, 1) + (B, 1, 20) - (B, 2020, 20) = (B, 2020, 20)
        overlaps = iw * ih / ua # ! (B, 2020, 20) * (B, 2020, 20) / (B, 2020, 20) = (B, 2020, 20)

        # mask the overlap here.
        # ! 이전에 식별한 비정상적인 gt_boxes, rois 처리
        # ! gt_area의 크기가 0인 경우 0으로, anchor_area가 0인 경우 -1로 처리
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

        # ! overlaps = (B, 2020, 20)

    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps
