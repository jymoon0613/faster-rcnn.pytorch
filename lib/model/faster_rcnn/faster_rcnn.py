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
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model) # self.dout_base_model = 512
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes) # 21 (20 category + 1 bg)

        # cfg.POOLING_SIZE = 7
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        # 참고: checker.ipynb
        ## im_data   = input image                   (B, 3, H, W) -> 2D images
        ## im_info   = image information             (B, 3)       -> (h, w, ?)
        ## gt_boxes  = ground-truth bboxes           (B, 20, 5)   -> (x1, y1, x2, y2, class_label)
        ## num_boxes = number of ground-truth bboxes (B,)         -> (#bboxes)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # im_data: (B, 3, 600, 600)
        ## 입력 resolution이 (600, 600)이라고 가정함
        base_feat = self.RCNN_base(im_data)
        # base_feat: (B, 512, 37, 37)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # rois          = (B, 2000, 5) -> 2000개의 proposals가 (batch_num, x1, y1, x2, y2)의 형태로 저장되어있음
        # rpn_loss_cls  = scalar value -> RPN의 cls_loss
        # rpn_loss_bbox = scalar value -> RPN의 bbox_loss

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # detection loss 계산을 위한 target 생성
            # rois      = (B, 2000, 5)
            # gt_boxes  = (B, 20, 5)
            # num_boxes = (B,)
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois            = (B, 128, 5)
            # rois_label      = (B, 128)
            # rois_target     = (B, 128, 4)
            # rois_inside_ws  = (B, 128, 4)
            # rois_outside_ws = (B, 128, 4)

            rois_label = Variable(rois_label.view(-1).long()) # (128B)                           
            rois_target = Variable(rois_target.view(-1, rois_target.size(2))) # (128B, 4)  
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2))) # (128B, 4)
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2))) # (128B, 4)
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        # RoI pooling 혹은 RoI align을 적용하여 feature 추출
        # model.roi_layers.roi_pool의 SlowROIPool 참조
        # RoI align은 RoI pooling과 달리 bilinear interpolation을 사용하여 feature maps 상에서 bbox의 더 정확한 위치에서 features를 추출함
        # base_feat = (B, 512, 37, 37)
        # rois.view(-1, 5) = (128B, 5)
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            # pooled_feat (128B, 512, 7, 7)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            # pooled_feat (128B, 512, 7, 7)

        # feed pooled features to top model
        # features를 detection head로 전달하고 loss 계산
        # pooled_feat (128B, 512, 7, 7)
        pooled_feat = self._head_to_tail(pooled_feat)
        # pooled_feat = (128B, 4096)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        # bbox_pred = (128B, 4)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) # bbox_pred_view = (128B, 1, 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)) # (128B, 1, 4)
            bbox_pred = bbox_pred_select.squeeze(1) # (128B, 4)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat) # (128B, 21)
        cls_prob = F.softmax(cls_score, 1) # (128B, 21)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training: # True
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) # scalar loss value

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws) # scalar loss value


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1) # (B, 128, 21)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1) # (B, 128, 4)

        # rois           = (B, 128, 5)
        # cls_prob       = (B, 128, 21)
        # bbox_pred      = (B, 128, 4)
        # rpn_loss_cls   = scalar loss value
        # rpn_loss_bbox  = scalar loss value
        # RCNN_loss_cls  = scalar loss value
        # RCNN_loss_bbox = scalar loss value
        # rois_label     = (B, 128)
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
