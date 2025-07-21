import cv2
import numpy as np
import torch
from utils.dense_pose.image_vis import BoxMode , DensePoseChartResultWithConfidences,DensePoseResultsVisualizer
import torch.nn.functional as F



def make_int_box(box):
    int_box = [0, 0, 0, 0]
    int_box[0], int_box[1], int_box[2], int_box[3] = tuple(box.long().tolist())
    return int_box[0], int_box[1], int_box[2], int_box[3]



def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm,
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        * (coarse_segm_bbox > 0).long()
    )
    return labels


def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv



class Saver():

    @classmethod
    def save_denseimage(self,entry, instances,out_path,image_name):
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        if instances["pred_densepose"] and instances["pred_boxes"]:
            dpout = instances["pred_densepose"]
            boxes_xyxy = instances["pred_boxes"]
            boxes_xywh = self.extract_boxes_xywh_from_instances(instances)
            
            results = [self.densepose_chart_predictor_output_to_result_with_confidences(dpout, boxes_xyxy["tensor"][[i]]) for i in range(dpout["coarse_segm"].size(0))]
        
            image_label ,image_cmap = DensePoseResultsVisualizer().visualize(image, (results,boxes_xywh))
            
            cv2.imwrite('{}/{}_vis.jpg'.format(out_path, image_name), image_label)
            cv2.imwrite('{}/{}.jpg'.format(out_path, image_name), image_cmap)
            return image_label ,image_cmap
            
        else:
            return None, None
        
    @classmethod    
    def extract_boxes_xywh_from_instances(self,instances, select=None):
        if instances["pred_boxes"] :
            boxes_xywh = instances["pred_boxes"]["tensor"].clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]
            return boxes_xywh if select is None else boxes_xywh[select]
        return None  
    
    @classmethod
    def densepose_chart_predictor_output_to_result_with_confidences(self,predictor_output, boxes):
   
    
        assert len([predictor_output["coarse_segm"]]) == 1 and len(boxes) == 1, (
            f"Predictor output to result conversion can operate only single outputs"
            f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
        )
        boxes_xyxy_abs = boxes.clone()
        boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, 0, 1)
        box_xywh = make_int_box(boxes_xywh_abs[0])

        labels = resample_fine_and_coarse_segm_tensors_to_bbox(predictor_output["fine_segm"],predictor_output["coarse_segm"],box_xywh).squeeze(0)
        
        uv = resample_uv_tensors_to_bbox(predictor_output["u"],predictor_output["v"],labels,box_xywh)

        return DensePoseChartResultWithConfidences(labels=labels, uv=uv)  