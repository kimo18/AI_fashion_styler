import os
import onnxruntime as ort
from utils.humanparse_preprocess import ImageReader
from utils.landmark_json import Landmark_Json
from utils.cloth_mask import ClothMask
from utils.get_parse_agnostic import Parse_Agnostic
from PIL import Image
import cv2
import numpy as np
data_path = os.path.join(".","Data")
    
image_dir = os.path.join(".","Data","image")
clothes_dir = os.path.join(".","Data","cloth")

human_parsing_dir = os.path.join(".","Data","image-parse-v3")
human_parsing_agnosti_dir = os.path.join(".","Data","image-parse-agnostic-v3.2")

clothes_dir = os.path.join(".","Data","cloth")
clothes_mask_dir = os.path.join(".","Data","cloth-mask")

openpose_json_dir = os.path.join(".","Data","openpose_json")
onnx_dir = "onnx"

image_path = "Data/image/front.jpg"


# Load the image with OpenCV (loads as BGR by default)
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Resize image to 800x800 (width x height)
resized_img = cv2.resize(img, (800, 800)).astype(np.float32)

# Optional: normalize to 0-1 range (some models expect this)
# img_float /= 255.0  # Note: size is (width, height)

# Convert to channels-first format: [C, H, W]
img_chw = np.transpose(resized_img, (2, 0, 1))
# img_chw = np.expand_dims(img_chw, axis=0)

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(os.path.join(".",onnx_dir,"denseparse","densepose_full.onnx"),sess_options=options, providers=['CPUExecutionProvider'])


input_names = [inp.shape for inp in session.get_inputs()]
output_names = [out.shape for out in session.get_outputs()]

print("Inputs:", input_names)
print("Outputs:", output_names)


input_dict = {
    'input': img_chw               #1,3,612,408
}
x,y,q,e,g,f,d= session.run(None, input_dict)


print(x.shape,y.shape,q.shape,e.shape,g.shape,f.shape,d.shape)



