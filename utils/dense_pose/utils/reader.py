from PIL import Image
import numpy as np
import os
from tqdm import tqdm
_EXIF_ORIENT = 274  # exif 'Orientation' tag



class ImageReader():
    
    @classmethod
    def read_image(self,file_name, format=None):

        images = []
        files = [f for f in os.listdir(file_name) if f.lower().endswith(('.jpg'))]
        for name in tqdm(files):
            path = os.path.join(file_name, name)
            image = Image.open(path)      
            image = self._apply_exif_orientation(image)
            images.append(self.convert_PIL_to_numpy(image, format))
        return images
    @classmethod
    def _apply_exif_orientation(self,image):

        if not hasattr(image, "getexif"):
            return image

        try:
            exif = image.getexif()
        except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
            exif = None

        if exif is None:
            return image

        orientation = exif.get(_EXIF_ORIENT)

        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)

        if method is not None:
            return image.transpose(method)
        return image
    @classmethod
    def convert_PIL_to_numpy(self,image, format):

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC

        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        return image


