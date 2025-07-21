from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional
import numpy as np
import torch
import math
import cv2 
import logging
N_PART_LABELS = 24

class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box, from_mode: "BoxMode", to_mode: "BoxMode"):
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr
        


@dataclass
class DensePoseChartResultWithConfidences:
    """
    We add confidence values to DensePoseChartResult
    Thus the results are represented by two tensors:
    - labels (tensor [H, W] of long): contains estimated label for each pixel of
        the detection bounding box of size (H, W)
    - uv (tensor [2, H, W] of float): contains estimated U and V coordinates
        for each pixel of the detection bounding box of size (H, W)
    Plus one [H, W] tensor of float for each confidence type
    """

    labels: torch.Tensor
    uv: torch.Tensor
    sigma_1: Optional[torch.Tensor] = None
    sigma_2: Optional[torch.Tensor] = None
    kappa_u: Optional[torch.Tensor] = None
    kappa_v: Optional[torch.Tensor] = None
    fine_segm_confidence: Optional[torch.Tensor] = None
    coarse_segm_confidence: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device, except if their value is None
        """

        def to_device_if_tensor(var: Any):
            if isinstance(var, torch.Tensor):
                return var.to(device)
            return var

        return DensePoseChartResultWithConfidences(
            labels=self.labels.to(device),
            uv=self.uv.to(device),
            sigma_1=to_device_if_tensor(self.sigma_1),
            sigma_2=to_device_if_tensor(self.sigma_2),
            kappa_u=to_device_if_tensor(self.kappa_u),
            kappa_v=to_device_if_tensor(self.kappa_v),
            fine_segm_confidence=to_device_if_tensor(self.fine_segm_confidence),
            coarse_segm_confidence=to_device_if_tensor(self.coarse_segm_confidence),
        )        



class MatrixVisualizer:
    """
    Base visualizer for matrix data
    """

    def __init__(
        self,
        inplace=True,
        cmap=cv2.COLORMAP_PARULA,
        val_scale=1.0,
        alpha=0.7,
        interp_method_matrix=cv2.INTER_LINEAR,
        interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        # if self.inplace:
        image_target_bgr = image_bgr
        # else:
        image_target_labels = np.zeros((image_bgr.shape[0],image_bgr.shape[1]))
        image_target_cmap = np.zeros((image_bgr.shape))

        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_labels = mask > 0
        mask_cmap = np.tile((mask>0)[:, :, np.newaxis], [1, 1, 3])
        mask_bg = np.tile((mask==0)[:, :, np.newaxis], [1, 1, 3])

        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
                
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        # matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_labels[y : y + h, x : x + w][mask_labels] =  matrix_scaled_8u[mask_labels]
        image_target_cmap[y : y + h, x : x + w][mask_cmap] =  matrix_vis[mask_cmap]

        matrix_vis[mask_bg] = image_target_bgr[y : y + h, x : x + w, :][mask_bg]
        image_target_bgr[y : y + h, x : x + w, :] = (
            image_target_bgr[y : y + h, x : x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )

        
        return image_target_labels.astype(np.uint8), image_target_cmap.astype(np.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == np.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == np.uint8





def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]




@dataclass
class DensePoseResultsVisualizer:
    
    def visualize(
        self,
        image_bgr,
        results_and_boxes_xywh
    ):
       
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.cpu().numpy()
        context = image_bgr
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).type(torch.uint8)
            image_label, image_cmap = self.visualize_iuv_arr(context, iuv_array.cpu().numpy(), boxes_xywh[i])
            return image_label ,image_cmap
        # image_bgr = context
        


    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        # image_bgr = context
        matrix = _extract_i_from_iuvarr(iuv_arr)
        segm = _extract_i_from_iuvarr(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_labels, image_cmap = MatrixVisualizer(val_scale = 255/N_PART_LABELS).visualize(context, mask, matrix, bbox_xywh)
        return image_labels, image_cmap