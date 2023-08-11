# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import numpy as np

from model import _C


class _ROIPool(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        output, argmax = _C.roi_pool_forward(
            input, roi, spatial_scale, output_size[0], output_size[1]
        )
        ctx.save_for_backward(input, roi, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_pool_backward(
            grad_output,
            input,
            rois,
            argmax,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
        )
        return grad_input, None, None, None


roi_pool = _ROIPool.apply


class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr

class SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        # ! input feature map의 크기와 output feature map의 크기를 고려하여 stride와 kernel의 크기를 조정하는 adaptive max pooling을 사용
        # ! output size = (cfg.POOLING_SIZE, cfg.POOLING_SIZE) = (7, 7)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size) 
        self.size = output_size # ! (7, 7)

    def forward(self, images, rois, roi_idx):

        # ! images = (B, 512, 37, 37)
        # ! rois   = (128B, 5)
        # ! 현재 구현된 예시 함수에서 rois의 좌표는 입력 이미지의 크기에 따라 0-1사이의 값으로 normalize되어 있다고 가정함

        n = rois.shape[0] # ! 128B
        h = images.size(2) # ! 37
        w = images.size(3) # ! 37
        x1 = rois[:,0] # ! (128B,)
        y1 = rois[:,1] # ! (128B,)
        x2 = rois[:,2] # ! (128B,)
        y2 = rois[:,3] # ! (128B,)

        # ! 주어진 feature maps의 h, w에 따라 x1, y1, x2, y2를 곱하여 비율에 맞게 scaling함
        # ! feature maps 상에서의 좌표로 변환됨
        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)
        
        res = []
        # ! 모든 rois에 대해 반복
        for i in range(n):
            # ! 해당 roi에 대응되는 feature maps 선택 (1, 512 ,37, 37)
            img = images[roi_idx[i]].unsqueeze(0) 

            # ! 선택된 feature maps에서 bbox region 선택 (1, 512 ,(y2-y1), (x2-x1))
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]] 

            # ! maxpooling 적용 (1, 512, 7, 7)
            img = self.maxpool(img) 
            res.append(img) # ! 결과 저장
        res = torch.cat(res, dim=0)

        # ! (128B, 512, 7, 7)
        return res