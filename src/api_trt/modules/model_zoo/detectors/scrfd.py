# -*- coding: utf-8 -*-
# Based on Jia Guo reference implementation at
# https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py


from __future__ import division
import time
from typing import Union
from functools import wraps
import logging

import cv2
import numpy as np
from numba import njit

from .common.nms import nms
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    import cupy as cp
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except BaseException:
    DIT = None


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t0 = time.time()
        result = f(*args, **kw)
        took_ms = (time.time() - t0) * 1000
        logging.debug(f'func: "{f.__name__}" took: {took_ms:.4f} ms')
        return result

    return wrap


@njit(fastmath=True, cache=True)
def distance2bbox(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    # WARNING! Don't try this at home, without Numba at least...
    # Here we use C-style function instead of Numpy matrix operations
    # since after Numba compilation code seems to work 2-4x times faster.

    for ix in range(0, distance.shape[0]):
        distance[ix, 0] = points[ix, 0] - distance[ix, 0]
        distance[ix, 1] = points[ix, 1] - distance[ix, 1]
        distance[ix, 2] = points[ix, 0] + distance[ix, 2]
        distance[ix, 3] = points[ix, 1] + distance[ix, 3]

    return distance


@njit(fastmath=True, cache=True)
def distance2kps(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    # WARNING! Don't try this at home, without Numba at least...
    # Here we use C-style function instead of Numpy matrix operations
    # since after Numba compilation code seems to work 2-4x times faster.

    for ix in range(0, distance.shape[1], 2):
        for j in range(0, distance.shape[0]):
            distance[j, ix] += points[j, 0]
            distance[j, ix + 1] += points[j, 1]

    return distance


@njit(fastmath=True, cache=True)
def fast_multiply(matrix, value):
    """
    Fast multiply list on value

    :param matrix: List of values
    :param value: Multiplier
    :return: Multiplied list
    """
    for ix in range(0, matrix.shape[0]):
        matrix[ix] *= value
    return matrix


@njit(fastmath=True, cache=True)
def single_distance2bbox(point, distance):
    """
    Fast conversion of single bbox distances to coordinates

    :param point: Anchor point
    :param distance: Bbox distances from anchor point
    :return: bbox
    """
    distance[0] = point[0] - distance[0]
    distance[1] = point[1] - distance[1]
    distance[2] = point[0] + distance[2]
    distance[3] = point[1] + distance[3]
    return distance


@njit(fastmath=True, cache=True)
def single_distance2kps(point, distance):
    """
    Fast conversion of single keypoint distances to coordinates

    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :return: keypoint
    """
    for ix in range(0, distance.shape[0], 2):
        distance[ix] += point[0]
        distance[ix + 1] += point[1]
    return distance


@njit(fastmath=True, cache=True)
def generate_proposals(score_blob, bbox_blob, kpss_blob, stride, anchors, threshold, score_out, bbox_out, kpss_out,
                       offset):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated np.ndarrays for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores np.ndarray
    :param bbox_out: Output bbox np.ndarray
    :param kpss_out: Output key points np.ndarray
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset

    for ix in range(0, anchors.shape[0]):
        if score_blob[ix, 0] > threshold:
            bbox = fast_multiply(bbox_blob[ix], stride)
            kpss = fast_multiply(kpss_blob[ix], stride)
            bbox = single_distance2bbox(anchors[ix], bbox)
            kpss = single_distance2kps(anchors[ix], kpss)
            score_out[total] = score_blob[ix]
            bbox_out[total] = bbox
            kpss_out[total] = kpss
            total += 1

    return score_out, bbox_out, kpss_out, total


# @timing
@njit(fastmath=True, cache=True)
def filter(bboxes_list: np.ndarray, kpss_list: np.ndarray,
           scores_list: np.ndarray):
    """
    Filter postprocessed network outputs with NMS

    :param bboxes_list: List of bboxes (np.ndarray)
    :param kpss_list: List of keypoints (np.ndarray)
    :param scores_list: List of scores (np.ndarray)
    :return: Face bboxes with scores [t,l,b,r,score], and key points
    """

    scores_ravel: np.ndarray = scores_list.reshape(-1)
    order: np.ndarray = scores_ravel.argsort()[::-1]
    pre_det = np.hstack((bboxes_list, scores_list))
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    keep = np.asarray(keep)
    det = pre_det[keep, :]
    kpss = kpss_list.reshape((kpss_list.shape[0], -1, 2))
    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]

    return det, kpss


def _normalize_on_device(input, stream, out):
    """
    Normalize image on GPU using inference backend preallocated buffers

    :param input: Raw image as nd.ndarray with HWC shape
    :param stream: Inference backend CUDA stream
    :param out: Inference backend pre-allocated input buffer
    :return: Image shape after preprocessing
    """
    input = np.expand_dims(input, axis=0)
    allocate_place = np.prod(input.shape)
    with stream:
        g_img = cp.asarray(input)
        g_img = g_img[..., ::-1]
        g_img = cp.transpose(g_img, (0, 3, 1, 2))
        g_img = cp.subtract(g_img, 127.5, dtype=cp.float32)
        out.device[:allocate_place] = cp.multiply(g_img, 1 / 128).flatten()
    return g_img.shape


class SCRFD:

    def __init__(self, inference_backend: Union[DIT, DIO], ver=1):
        self.session = inference_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.ver = ver
        self.out_shapes = None
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.stream = None
        self.input_ptr = None

    def prepare(self, nms_treshold: float = 0.45, **kwargs):
        """
        Read network params and populate class parameters

        :param nms_treshold: Threshold for NMS IoU

        """
        self.nms_threshold = nms_treshold
        self.session.prepare()
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

        # Preallocate reusable arrays for proposals
        max_prop_len = self._get_max_prop_len(self.input_shape,
                                              self._feat_stride_fpn,
                                              self._num_anchors)
        self.score_list = np.zeros((max_prop_len, 1))
        self.bbox_list = np.zeros((max_prop_len, 4))
        self.kpss_list = np.zeros((max_prop_len, 10))

        # Check if exec backend provides CUDA stream
        try:
            self.stream = self.session.stream
            self.input_ptr = self.session.input_ptr
        except BaseException:
            pass

    # @timing
    def detect(self, img, threshold=0.5):
        """
        Run detection pipeline for provided image

        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """

        input_height = img.shape[0]
        input_width = img.shape[1]
        blob = self._preprocess(img)
        net_outs = self._forward(blob)

        bboxes_list, kpss_list, scores_list = self._postprocess(net_outs, input_height, input_width, threshold)

        det, kpss = filter(
            bboxes_list, kpss_list, scores_list)

        return det, kpss

    @staticmethod
    def _get_max_prop_len(input_shape, feat_strides, num_anchors):
        """
        Estimate maximum possible number of proposals returned by network

        :param input_shape: maximum input shape of model (i.e (1, 3, 640, 640))
        :param feat_strides: model feature strides (i.e. [8, 16, 32])
        :param num_anchors: model number of anchors (i.e 2)
        :return:
        """

        ln = 0
        pixels = input_shape[2] * input_shape[3]
        for e in feat_strides:
            ln += pixels / (e * e) * num_anchors
        return int(ln)

    # @timing
    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            centers.append(anchor_centers)
        return centers

    # @timing
    def _preprocess(self, img):
        """
        Normalize image on CPU if backend can't provide CUDA stream,
        otherwise preprocess image on GPU using CuPy

        :param img: Raw image as np.ndarray with HWC shape
        :return: Preprocessed image or None if image was processed on device
        """

        blob = None
        if self.stream:
            self.infer_shape = _normalize_on_device(
                img, self.stream, self.input_ptr)
        else:
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(
                img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        return blob

    def _forward(self, blob):
        """
        Send input data to inference backend.

        :param blob: Preprocessed image of shape NCHW or None
        :return: network outputs
        """

        t0 = time.time()
        if self.stream:
            net_outs = self.session.run(
                from_device=True, infer_shape=self.input_shape)
        else:
            net_outs = self.session.run(blob)
        t1 = time.time()
        logging.debug(f'Inference cost: {(t1 - t0) * 1000:.3f} ms.')
        return net_outs

    # @timing
    def _postprocess(self, net_outs, input_height, input_width, threshold):
        """
        Precompute anchor points for provided image size and process network outputs

        :param net_outs: Network outputs
        :param input_height: Input image height
        :param input_width: Input image width
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        key = (input_height, input_width)

        if not self.center_cache.get(key):
            self.center_cache[key] = self._build_anchors(input_height, input_width, self._feat_stride_fpn,
                                                         self._num_anchors)
        anchor_centers = self.center_cache[key]
        bboxes, kpss, scores = self._process_strides(net_outs, threshold, anchor_centers)
        return bboxes, kpss, scores

    def _process_strides(self, net_outs, threshold, anchor_centers):
        """
        Process network outputs by strides and return results proposals filtered by threshold

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """

        offset = 0

        for idx, stride in enumerate(self._feat_stride_fpn):
            score_blob = net_outs[idx][0]
            bbox_blob = net_outs[idx + self.fmc][0]
            kpss_blob = net_outs[idx + self.fmc * 2][0]
            stride_anchors = anchor_centers[idx]
            self.score_list, self.bbox_list, self.kpss_list, total = generate_proposals(score_blob, bbox_blob,
                                                                                        kpss_blob, stride,
                                                                                        stride_anchors, threshold,
                                                                                        self.score_list, self.bbox_list,
                                                                                        self.kpss_list, offset)
            offset = total
        return self.bbox_list[:offset], self.kpss_list[:offset], self.score_list[:offset]
