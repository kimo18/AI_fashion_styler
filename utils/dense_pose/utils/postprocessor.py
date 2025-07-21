import torch
from PIL import Image
import numpy as np

class PreProcessor():
    
    @classmethod
    def pre_process(self,original_image,NewH, New_W):
        height, width = original_image.shape[:2]
        image = self.apply_image(original_image,height, width,NewH, New_W)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to("cpu")
        return [{"image": image, "height": height, "width": width}] 
    
    @classmethod    
    def apply_image(self, img, h, w, newh, neww, interp=Image.BILINEAR):
        assert img.shape[:2] == (h, w)
        assert len(img.shape) <= 4
        interp_method = interp 

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((neww, newh), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (newh, neww), mode=mode, align_corners=align_corners
            )
            shape[:2] = (newh,neww)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret






class PostProcess():

    @classmethod
    def _postprocess(self,instances, batched_inputs, image_sizes ):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = self.detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    @classmethod    
    def detector_postprocess(
    self,results, output_height: int, output_width: int, mask_threshold: float = 0.5
    ):

        if isinstance(output_width, torch.Tensor):
            # This shape might (but not necessarily) be tensors during tracing.
            # Converts integer tensors to float temporaries to ensure true
            # division is performed when computing scale_x and scale_y.
            output_width_tmp = output_width.float()
            output_height_tmp = output_height.float()
            new_size = torch.stack([output_height, output_width])
        else:
            new_size = (output_height, output_width)
            output_width_tmp = output_width
            output_height_tmp = output_height



        
        scale_x, scale_y = (
            output_width_tmp / results["image_shape"][1],
            output_height_tmp / results["image_shape"][0],
        )
        # results = Instances(new_size, **results.get_fields())

        if results["pred_boxes"]:
            output_boxes = results["pred_boxes"]["tensor"]
        elif results["proposal_boxes"]:
            output_boxes = results["proposal_boxes"]
        else:
            output_boxes = None
        assert output_boxes is not None, "Predictions must contain boxes!"
        output_boxes = self.scale(output_boxes,scale_x, scale_y)
        output_boxes = self.clip(output_boxes,new_size)
        keep = self.nonempty(output_boxes)
        results = self.update_dict(results,keep)
        return results
    
    @classmethod
    def scale(self,tensor, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        tensor[:, 0::2] *= scale_x
        tensor[:, 1::2] *= scale_y
        return tensor

    @classmethod
    def clip(self,tensor,  box_size) -> None:
            """
            Clip (in place) the boxes by limiting x coordinates to the range [0, width]
            and y coordinates to the range [0, height].

            Args:
                box_size (height, width): The clipping box's size.
            """
            assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
            h, w = box_size
            x1 = tensor[:, 0].clamp(min=0, max=w)
            y1 = tensor[:, 1].clamp(min=0, max=h)
            x2 = tensor[:, 2].clamp(min=0, max=w)
            y2 = tensor[:, 3].clamp(min=0, max=h)
            tensor = torch.stack((x1, y1, x2, y2), dim=-1)
            return tensor

    @classmethod
    def nonempty(self,tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep 

    @classmethod    
    def update_dict(self,d,keep):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                # Example update: convert tensor to float32
                d[k] = d[k][keep]
            elif isinstance(v, dict):
                # Recursive call if value is another dict
                self.update_dict(v,keep)
            # You can add more types if needed
        return d

