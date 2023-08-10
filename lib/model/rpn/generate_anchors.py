from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    
    # ! base_size = 16
    # ! ratios    = [0.5,1,2]
    # ! scales    = [8,16,32]

    # ! 기본 anchor 정의
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # ! base_anchor = [0, 0, 15, 15]

    # ! base_anchor에 대해 서로 다른 aspect_ratio의 anchor boxes 생성
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # ! ratio_anchors = (3, 4)

    # ! 생성된 ratio_anchors에 대해 서로 다른 scale의 anchor boxes를 다시 생성
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    # ! anchors = 
    # ! [
    # !  [x1-w1-s1, y1-h1-s1, x2-w1-s1, y2-h1-s1],
    # !  [x1-w1-s2, y1-h1-s2, x2-w1-s2, y2-h1-s2],
    # !  [x1-w1-s3, y1-h1-s3, x2-w1-s3, y2-h1-s3],
    # !  [x1-w2-s1, y1-h2-s1, x2-w2-s1, y2-h2-s1],
    # !  [x1-w2-s2, y1-h2-s2, x2-w2-s2, y2-h2-s2], 
    # !  [x1-w2-s3, y1-h2-s3, x2-w2-s3, y2-h2-s3],
    # !  [x1-w3-s1, y1-h3-s1, x2-w3-s1, y2-h3-s1],
    # !  [x1-w3-s2, y1-h3-s2, x2-w3-s2, y2-h3-s2], 
    # !  [x1-w3-s3, y1-h3-s3, x2-w3-s3, y2-h3-s3],
    # ! ]
    # ! (9, 4)

    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # ! 주어진 anchor의 너비/높이 (w, h), 중심좌표 (x_ctr, y_ctr) 계산

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    # ! ws    = [w1, w2, w3]
    # ! hs    = [h1, h2, h3]
    # ! x_ctr = 중심 x 좌표
    # ! y_ctr = 중심 y 좌표

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # ! ws = [[w1], 
    # !       [w2],
    # !       [w3]]
    # ! hs = [[h1], 
    # !       [h2],
    # !       [h3]]

    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),  # ! 7.5 - 0.5 * ([[w1], [w2], [w3]] - 1) = top-left corner의 x좌표:     x1s = [[x1-w1], [x1-w2], [x1-w3]]
                         y_ctr - 0.5 * (hs - 1),  # ! 7.5 - 0.5 * ([[h1], [h2], [h3]] - 1) = top-left corner의 y좌표:     y1s = [[y1-h1], [y1-h2], [y1-h3]]
                         x_ctr + 0.5 * (ws - 1),  # ! 7.5 + 0.5 * ([[w1], [w2], [w3]] - 1) = bottom-right corner의 x좌표: x2s = [[x2-w1], [x2-w2], [x2-w3]]
                         y_ctr + 0.5 * (hs - 1))) # ! 7.5 + 0.5 * ([[h1], [h2], [h3]] - 1) = bottom-right corner의 y좌표: y2s = [[y2-h1], [y2-h2], [y2-h3]]
    
    # ! anchors = 
    # ! [
    # !  [x1-w1, y1-h1, x2-w1, y2-h1],
    # !  [x1-w2, y1-h2, x2-w2, y2-h2],
    # !  [x1-w3, y1-h3, x2-w3, y2-h3],
    # ! ]
    # ! (3, 4)

    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    # ! 주어진 anchor의 너비/높이 (w, h), 중심좌표 (x_ctr, y_ctr) 계산
    w, h, x_ctr, y_ctr = _whctrs(anchor) # ! 16, 16, 7.5, 7.5

    # ! 주어진 anchor의 크기 계산
    size = w * h # ! 16 x 16 = 256

    # ! 서로 다른 aspect ratio로 boradcasting
    size_ratios = size / ratios # ! 256 / [0.5,1,2] = [512., 256., 128.]

    # ! broadcasting된 aspect_ratios 각각의 너비와 높이 계산
    ws = np.round(np.sqrt(size_ratios)) # ! [23., 16., 11.]
    hs = np.round(ws * ratios) # ! [12., 16., 22.]

    # ! 계산 결과를 바탕으로 서로 다른 aspect ratios의 anchor bboxes 생성
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # ! anchors = (3, 4)

    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    # ! 주어진 anchor box의 너비, 높이, 중심좌표 계산
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # ! anchor box의 너비를 서로 다른 scale ratio로 boradcasting
    ws = w * scales # ! w * [8, 16, 32] = [8w, 16w, 32w]

    # ! anchor box의 높이를 서로 다른 scale ratio로 boradcasting
    hs = h * scales # ! h * [8, 16, 32] = [8h, 16h, 32h]

    # ! 계산 결과를 바탕으로 anchor bboxes 생성
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    # ! anchors = (3, 4)

    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
