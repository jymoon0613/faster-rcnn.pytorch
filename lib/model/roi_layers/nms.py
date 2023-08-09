# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from model import _C
import numpy as np
import torch

nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

def nms_cpu(dets, thresh):

    # dets   = (12000, 5) (dets[4]가 해당 proposal의 objectness score를 의미함)
    # thresh = 0.7

    dets = dets.numpy()
    x1 = dets[:, 0] # (12000,)
    y1 = dets[:, 1] # (12000,)
    x2 = dets[:, 2] # (12000,)
    y2 = dets[:, 3] # (12000,)
    scores = dets[:, 4] # (12000,)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # proposal의 넓이 계산 # (12000,)
    order = scores.argsort()[::-1] # 현재 nms 함수에서는 score가 정렬되어 있다고 가정하지 않음. 따라서 score를 정렬
    # order = objectness scores를 기준으로 내림차순으로 정렬했을 때의 proposal indices (12000,)

    keep = [] # 유지할 proposal indices를 저장할 리스트
    while order.size > 0:
        i = order.item(0) # 현재 indices에서 가장 첫번째(= 현재 가장 objectness score가 높은 proposal의) index를 추출
        keep.append(i) # 해당 index를 저장
        
        # NMS를 수행하기 위해 IoU를 계산해야 함 
        # 먼저, 현재 선택된 index의 proposal과 다른 모든 proposals의 intersection을 계산
        # intersection 계산을 위해 intersection region의 (xx1, yy1, xx2, yy2) 좌표 계산
        # 이때 np.maximum 연산은 변수 x1[i]와 리스트 x1[order[1:]]에 대해 수행되며, 이때 리스트 x1[order[1:]]의 각 요소는 변수 x1[i]와 비교했을 때 더 큰 값으로 유지/변경됨
        # 즉, xx1의 차원은 x1[order[1:]]의 차원과 동일함 (l이라고 칭함)
        # xx2와 yy2 계산에는 minimum이 쓰인다는 점을 주의
        xx1 = np.maximum(x1[i], x1[order[1:]]) # intersection region의 top-left corner x좌표: xx1 (l,)
        yy1 = np.maximum(y1[i], y1[order[1:]]) # intersection region의 top-left corner y좌표: yy1 (l,)
        xx2 = np.minimum(x2[i], x2[order[1:]]) # intersection region의 bottom-right corner x좌표: xx2 (l,)
        yy2 = np.minimum(y2[i], y2[order[1:]]) # intersection region의 bottom-right corner y좌표: yy2 (l,)

        # 계산된 (xx1, yy1, xx2, yy2)에 대해 intersection 계산을 위해 너비/높이 계산
        # 음수인 너비/높이는 0으로 clip함
        w = np.maximum(0.0, xx2 - xx1 + 1) # (l,)
        h = np.maximum(0.0, yy2 - yy1 + 1) # (l,)

        # 최종적으로 intersection 계산 
        inter = w * h # (l,)

        # IoU 계산
        # IoU = intersection / union
        # 선택된 proposal과 다른 모든 proposals 간의 IoU를 한번에 계산
        ovr = inter / (areas[i] + areas[order[1:]] - inter) # (l,)

        # 계산된 IoU에 기반하여 IoU가 IoU threshold보다 작은 proposals만 유지함
        # 왜? 선택된 i번째 proposal은 현재 가장 objectness score가 높은 proposal임
        # 그렇기 때문에 현재 선택된 proposal과 IoU threshold보다 크게 겹치는 proposals는 같은 object를 capture하면서 objectness scores는 더 낮은 proposals임
        # 따라서 그러한 proposals는 더 이상 고려할 필요가 없으므로 제거함
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    # 더 이상 고려할 proposals가 없을 때까지 반복

    return torch.IntTensor(keep)
