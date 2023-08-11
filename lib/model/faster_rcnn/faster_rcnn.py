import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes) # ! 21 (20 obj + 1 bg)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # ! RPN 정의
        # ! self.RCNN_rpn = _RPN(512)
        self.RCNN_rpn = _RPN(self.dout_base_model)

        # ! Faster R-CNN의 training target을 생성하는 layer 정의
        # ! self.RCNN_proposal_target = _ProposalTargetLayer(21)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # ! proposals feature 추출을 위한 RoI Pooling, RoI Align layer 정의
        # ! cfg.POOLING_SIZE = 7
        # ! self.RCNN_roi_pool = ROIPool((7, 7), 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        # ! self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):

        # ! im_data   = input image                   (B, 3, H, W) -> 2D images
        # ! im_info   = image information             (B, 3)       -> (h, w, scale)
        # ! gt_boxes  = ground-truth bboxes           (B, 20, 5)   -> (x1, y1, x2, y2, obj_class)
        # ! num_boxes = number of ground-truth bboxes (B,)         -> (#bboxes)

        batch_size = im_data.size(0) # ! B

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # ! VGG backbone으로부터 feature maps 추출
        # ! 이때 입력 resolution이 (600, 600)이라고 가정함
        # ! VGG-16 backbone의 경우 입력 이미지의 resoultion은 대략 16배 감소함
        # ! im_data = (B, 3, 600, 600)
        base_feat = self.RCNN_base(im_data)
        # ! base_feat = (B, 512, 37, 37)

        # ! Feature maps를 RPN에 입력하여 region proposals(= rois)를 생성함
        # ! RPN의 cls_loss, bbox_loss도 계산함
        # ! base_feat = (B, 512, 37, 37)
        # ! im_info   = (B, 3)
        # ! gt_boxes  = (B, 20, 5)
        # ! num_boxes = (B,)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # ! rois          = (B, 2000, 5)      -> 2000개의 region proposals가 (batch_num, x1, y1, x2, y2)의 형태로 저장되어있음
        # ! rpn_loss_cls  = scalar loss value -> RPN의 cls_loss
        # ! rpn_loss_bbox = scalar loss value -> RPN의 bbox_loss

        # if it is training phrase, then use ground trubut bboxes for refining
        # ! training 상황을 가정
        if self.training: # ! True
            # ! detection loss 계산을 위한 target 생성
            # ! rois      = (B, 2000, 5) -> 2000개의 region proposals가 (batch_num, x1, y1, x2, y2)의 형태로 저장되어있음
            # ! gt_boxes  = (B, 20, 5)   -> ground-truth bboxes
            # ! num_boxes = (B,)         -> number of ground-truth bboxes
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            # ! rois            = (B, 128, 5) -> 최종적으로 선택된 rois
            # ! rois_label      = (B, 128)    -> 각 rois에 대응되는 gt_cls_targets
            # ! rois_target     = (B, 128, 4) -> bbox_reg loss gt_bbox_reg_targets
            # ! rois_inside_ws  = (B, 128, 4) -> bbox_reg loss 가중치
            # ! rois_outside_ws = (B, 128, 4) -> bbox_reg loss 가중치 (inside와 동일)

            # ! reshape
            rois_label = Variable(rois_label.view(-1).long()) # ! (128B,)                           
            rois_target = Variable(rois_target.view(-1, rois_target.size(2))) # ! (128B, 4)  
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2))) # ! (128B, 4)
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2))) # ! (128B, 4)
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        # ! RoI pooling 혹은 RoI align을 적용하여 feature 추출
        # ! model.roi_layers.roi_pool의 SlowROIPool 참조
        # ! RoI align은 RoI pooling과 달리 bilinear interpolation을 사용하여 feature maps 상에서 bbox의 더 정확한 위치에서 features를 추출함
        # ! base_feat = (B, 512, 37, 37)
        # ! rois.view(-1, 5) = (128B, 5)
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            # ! pooled_feat (128B, 512, 7, 7)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            # ! pooled_feat (128B, 512, 7, 7)

        # feed pooled features to top model
        # ! features를 detection head로 전달
        # ! pooled_feat = (128B, 512, 7, 7)
        pooled_feat = self._head_to_tail(pooled_feat)
        # ! pooled_feat = fc7 = (128B, 4096)

        # compute bbox offset
        # ! pooled_feat을 Fast R-CNN bbox_reg branch에 입력
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # ! bbox_pred = (128B, 84) -> 21개의 class마다 (tx, ty, tw, th) 예측

        if self.training and not self.class_agnostic: # ! True and True
            # ! 각 bbox prediction을 class label에 대응되도록 정렬
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) # ! (128B, 21, 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)) # ! (128B, 1, 4)
            bbox_pred = bbox_pred_select.squeeze(1) # ! (128B, 4)

        # compute object classification probability
        # ! cls prediction 수행
        cls_score = self.RCNN_cls_score(pooled_feat) # ! (128B, 21)
        cls_prob = F.softmax(cls_score, 1) # ! (128B, 21)

        # ! 학습을 위한 loss 계산
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training: # ! True
            # ! classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) # ! scalar loss value

            # ! bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws) # ! scalar loss value

        # ! reshape
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1) # ! (B, 128, 21)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1) # ! (B, 128, 4)

        # ! rois           = (B, 128, 5)       -> 최종적으로 선택된 rois
        # ! cls_prob       = (B, 128, 21)      -> detection head의 class 예측값
        # ! bbox_pred      = (B, 128, 4)       -> detection head의 bbox offset 예측값
        # ! rpn_loss_cls   = scalar loss value -> RPN의 cls_loss 값
        # ! rpn_loss_bbox  = scalar loss value -> RPN의 bbox_reg_loss 값
        # ! RCNN_loss_cls  = scalar loss value -> detection head의 cls_loss 값
        # ! RCNN_loss_bbox = scalar loss value -> detection head의 bbox_reg_loss 값
        # ! rois_label     = (B, 128)          -> rois에 대한 gt_cls_labels
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
