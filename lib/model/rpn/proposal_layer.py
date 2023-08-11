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
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride # ! 16

        # ! anchor box 생성
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()
        # ! self._anchors = (9, 4) -> 서로 다른 aspect ratio, 서로 다른 scale의 anchor boxes에 대한 좌표 (x1, y1, x2, y2)

        self._num_anchors = self._anchors.size(0) # ! 9

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        # ! input[0] = rpn_cls_prob  = (B, 18, 37, 37) -> 모든 feature map positions, 모든 anchor boxes에 대한 objectness score 예측값
        # ! input[1] = rpn_bbox_pred = (B, 36, 37, 37) -> 모든 feature map positions, 모든 anchor boxes에 대한 bbox offset 예측값
        # ! input[2] = im_info       = (B, 3)
        # ! input[3] = cfg_key       = 'TRAIN'

        scores = input[0][:, self._num_anchors:, :, :] # ! 전체 cls scores 중 fg scores만을 추출 = (B, 9, 37, 37)
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # ! 12000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # ! 2000
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH     # ! 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE       # ! 8

        batch_size = bbox_deltas.size(0) # ! B

        feat_height, feat_width = scores.size(2), scores.size(3) # ! 37, 37

        # ! feature map의 각 좌표를 원본 이미지 상의 좌표로 변환
        shift_x = np.arange(0, feat_width) * self._feat_stride  # ! [0, 1, 2, ..., 37] * 16 -> [0, 16, 32, ..., 576] (37,)
        shift_y = np.arange(0, feat_height) * self._feat_stride # ! [0, 1, 2, ..., 37] * 16 -> [0, 16, 32, ..., 576] (37,)

        # ! 그리드 좌표로 변환
        shift_x, shift_y = np.meshgrid(shift_x, shift_y) # ! (37,37), (37,37)

        # ! bbox 형식으로 변환 -> [[x1, y1, x2, y2], ...]
        # ! 이때 x1=x2, y1=y2임 (특정 position을 의미하므로)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()
        # ! shifts = (1369, 4)

        A = self._num_anchors # ! 9
        K = shifts.size(0) # ! 37 * 37 = 1369

        self._anchors = self._anchors.type_as(scores)

        # ! 1369(37x37)의 모든 reference positions 상에서 9개의 anchor boxes를 정의함
        # ! 즉, anchors는 모든 reference positions 상에서 9개의 anchor boxes 좌표를 담고 있음 (1369, 9, 4)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # ! self._anchors = (9, 4)
        # ! self._anchors.view(1, A, 4) = (1, 9, 4)
        # ! shifts = (1369, 4)
        # ! shifts.view(K, 1, 4) = (1369, 1, 4)
        # ! broadcasting: anchors = (1, 9, 4) + (1369, 1, 4) = (1369, 9, 4)

        # ! reshape하고 batch size만큼 확장
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        # ! anchors = (B, 12321, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the ***same order as the anchors***:
        # ! anchors의 차원에 맞게 reshape
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous() # ! (B, 37, 37, 36)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4) # ! (B, 12321, 4)
        # ! bbox_deltas = (B, 12321, 4) -> 모든 positions, anchor boxes 별 bbox offset 예측값

        # Same story for the scores:
        # ! class score도 같은 방식으로 변경
        scores = scores.permute(0, 2, 3, 1).contiguous() # ! (B, 37, 37, 18)
        scores = scores.view(batch_size, -1) # ! (B, 12321)
        # ! scores = (B, 12321) -> 모든 positions, anchor boxes 별 objectness score 예측값

        # Convert anchors into proposals via bbox transformations
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
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        # ! proposals = (B, 12321, 4)
        # ! 출력된 proposals는 12321개의 초기 proposals를 담고 있음

        # 2. clip predicted boxes to image
        # ! 생성된 초기 proposals에서 이미지 영역 밖으로 벗어나는 boxes를 이미지 boundary로 clip해주는 처리를 수행
        proposals = clip_boxes(proposals, im_info, batch_size)
        # ! proposals = (B, 12321, 4)

        # ! 생성된 proposals 중 score 값을 기준으로 유의미한 것들만 선택하는 과정을 수행함
        scores_keep = scores # ! 원본 score 저장
        proposals_keep = proposals # ! 원본 proposals 저장
        _, order = torch.sort(scores_keep, 1, True) # ! objectness scores를 기준으로 내림차순으로 정렬
        # ! order = (B, 12321) -> 정렬된 objectness score indices 

        # ! 결과 저장할 tensor 정의
        output = scores.new(batch_size, post_nms_topN, 5).zero_() # ! (B, 2000, 5)
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            # ! i번째 batch의 모든 proposals (12321, 4)
            proposals_single = proposals_keep[i] 
            # ! i번째 batch의 모든 objectness scores (12321, )
            scores_single = scores_keep[i] 

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            # ! i번째 bacth의 정렬된 objectness score indices (12321, )
            order_single = order[i] 

            # ! NMS를 적용하기 전 가장 높은 scores를 갖는 proposals부터 일부만을 고려함
            # ! 현재 pre_nms_topN = 12000이므로 0보다 크고, 12321보다 작음
            # ! 따라서 score 상위 12000개의 proposals만을 고려함
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                # ! 상위 12000개 proposals의 indices (12000, )
                order_single = order_single[:pre_nms_topN]

            # ! 상위 12000개 proposals (12000, 4)
            proposals_single = proposals_single[order_single, :]
            # ! 상위 12000개 objectness score (12000, )
            scores_single = scores_single[order_single].view(-1,1) 

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            # ! proposals에 대해 objectness score를 기반으로 NMS 수행
            # ! roi_layers.nms의 nms_cpu 참고
            # ! nms_thresh = 0.7
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh) 
            keep_idx_i = keep_idx_i.long().view(-1)
            # ! keep_idx_i = (n,) -> NMS를 적용한 후 남아 있는 n개의 proposal indices 

            # ! NMS를 적용한 후 가장 높은 scores를 갖는 proposals 일부만을 고려함
            # ! 현재 post_nms_topN = 2000이므로 0보다 큼
            # ! 따라서 상위 2000개의 proposals만을 고려함
            # ! 이때 keep_idx_i는 NMS 과정에서 이미 objectness scores를 기준으로 내림차순 정렬되어 있음
            # ! 현재 예시에서 n은 2000보다 크다고 가정함
            if post_nms_topN > 0:
                # ! 상위 2000개 proposals의 indices
                keep_idx_i = keep_idx_i[:post_nms_topN] # ! (2000, # 상위 2000개 proposals
            # ! 상위 2000개 proposals
            proposals_single = proposals_single[keep_idx_i, :] # ! (2000, 4)
            # ! 상위 2000개 objectness score
            scores_single = scores_single[keep_idx_i, :] # ! (2000, )

            # ! 결과값 저장
            num_proposal = proposals_single.size(0) # ! 2000

            # ! (batch_index, x1, y1, x2, y2)로 구성됨
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        # ! output = (B, 2000, 5)

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
