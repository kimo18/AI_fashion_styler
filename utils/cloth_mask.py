import os 
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2

IMAGE_SIZE = 320
class ClothMask():
    def __init__(self,cloth_dir):
        self.cloth_dir = cloth_dir
        self.orig_size = None

    def preprocess(self,image: Image.Image):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    
    def postprocess(self,output_tensor, size):
        output_tensor = torch.from_numpy(output_tensor[0])
        mask = output_tensor.squeeze().detach().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        return binary

    # Morphological cleaning
    def clean_mask(self,mask_np):
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(closed)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                cv2.drawContours(cleaned, [cnt], 0, 255, -1)
        return cleaned


    def process_image(self):
        
        files = [f for f in os.listdir(self.cloth_dir) if f.lower().endswith(('.jpg'))]
        for name in tqdm(files):
            path = os.path.join(self.cloth_dir, name)
            image = Image.open(path).convert("RGB")
            self.orig_size = image.size
            input_tensor = self.preprocess(image).cpu().numpy()
            return input_tensor ,os.path.splitext(name)[0]