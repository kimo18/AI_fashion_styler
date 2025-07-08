import os
import numpy as np
import cv2
import random
from PIL import Image

IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)


def image_scaling(img):
    """
    Randomly scales the images between 0.5 to 2.0 times the original size using cv2.
    """
    # scale = random.uniform(0.5, 2.0)
    # h, w = img.shape[:2]
    # new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (408 , 612), interpolation=cv2.INTER_LINEAR)
    return img_resized


def random_crop_and_pad_image(image, crop_h, crop_w):
    """
    Randomly crop and pads the input image to size (crop_h, crop_w).
    Pads with zeros if needed.
    """
    h, w = image.shape[:2]

    # Pad if image is smaller than crop size
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))

    # Random crop
    h, w = image.shape[:2]
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)

    crop = image[top:top+crop_h, left:left+crop_w]
    return crop


def read_images(data_dir):
    jpg_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
    return [os.path.join(data_dir, f) for f in jpg_files] ,   [os.path.splitext(f)[0] for f in jpg_files]  


def preprocess_image(image_path, input_size=None,
                     random_scale=False,
                     random_mirror=False):
    """
    Load image from disk and apply preprocessing.
    
    Args:
        image_path: path to image file.
        input_size: tuple (height, width) to resize/crop image.
        random_scale: whether to randomly scale the image.
        random_mirror: whether to randomly mirror the image.
        
    Returns:
        preprocessed image as numpy array (H, W, 3), float32.
    """
    img = cv2.imread(image_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    
    img = image_scaling(img)

    if input_size is not None:
        h, w = input_size
        img = random_crop_and_pad_image(img, h, w)
    else:
        img = img.astype(np.float32)

    # Convert to float32 and subtract mean
    img = img.astype(np.float32)
    img -= IMG_MEAN

    return img



class ImageReader:
    """
    Reads and preprocesses images from a directory, no TensorFlow dependencies.
    """

    def __init__(self, data_dir, input_size=None,
                 random_scale=False,
                 random_mirror=False,
                 shuffle=False):
        self.data_dir = data_dir
        self.input_size = input_size
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.shuffle = shuffle
        self.label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

        self.image_paths,self.image_list = read_images(data_dir)
        
        if shuffle:
            random.shuffle(self.image_paths)

    def read_and_preprocess(self, index=0):
        """
        Returns one preprocessed image numpy array.
        """
        img_path = self.image_paths[index]
        img = preprocess_image(img_path, self.input_size,
                               self.random_scale, self.random_mirror)
        return img, self.image_list[index]

    def read_batch(self, batch_size=1):
        """
        Returns a batch of preprocessed images as numpy arrays stacked into
        shape (batch_size, H, W, 3).
        """
        batch = []
        for i in range(batch_size):
            idx = i % len(self.image_paths)
            img, image_name = self.read_and_preprocess(idx)
            batch.append(img)
        return np.stack(batch),image_name
    
    def decode_labels(self,mask, num_images=1, num_classes=21):
        """Decode batch of segmentation masks.
        
        Args:
        mask: result of inference after taking argmax.
        num_images: number of images to decode from the batch.
        num_classes: number of classes to predict (including background).
        
        Returns:
        A batch with num_images RGB images of the same size as the input. 
        """
        n, h, w, c = mask.shape
        assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
        outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
        for i in range(num_images):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < num_classes:
                        pixels[k_,j_] = self.label_colours[k]
            outputs[i] = np.array(img)
        return  Image.fromarray(outputs[0])



    