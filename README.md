# Fashion AI Setup and Usage Guide

## AI-Powered Personal Styling App

We’re developing an AI-powered personal styling app that helps you choose the perfect outfit with ease.

Our application allows you to virtually try on clothes by fitting outfits onto your own body image, so you can see exactly how they’ll look on you. It recommends styles tailored to your taste, helps you shop smartly, and saves you time by making fashion decisions faster and easier. All with the help of artificial intelligence.



This guide explains how to clone, set up, and run the project.

### Output Sequence Example

<p align="center">
  <img src="image.png" width="40%" /> 
  <img src="image2.png" width="40%" /> 
</p>

<p align="center">
  <strong>↓</strong>
</p>

<p align="center">
  <img src="image3.png" width="45%" /> 
</p>

<p align="center"><em>Result after applying our Model</em></p>

---

## Table of Contents

- [Setup Installation](#setup-installation)
- [Create and Activate Conda Environment](#create-and-activate-conda-environment)
- [Densepose Repository Installation](#densepose-repository-installation)
- [Install Dependencies](#install-dependencies)
- [Download Pretrained Model](#download-pretrained-model)
- [Fix Black Background Issue](#fix-black-background-issue)
- [Run DensePose Inference](#run-densepose-inference)

---



## Setup Installation

1. **Python Version**

   Make sure you have Python version **3.12.10** installed.

2. **Install Requirements**

   Install the required Python packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt

   ```
  Create a .env file in the project root directory and add your Groq API key like this: 

  ```bash
  GROQ_API_KEY="your_api_key"
  ```
3. **Download ONNX Folder**  
   Download the `onnx` folder from the following Google Drive link:
   [Link](https://drive.google.com/drive/folders/11wb58wtJfpQeAv7bqeS9v34EvO1ONEa_?usp=sharing)

4. **Place ONNX Folder**  
   After downloading, move the entire `onnx` folder into the **root directory** of your project (i.e., the same directory as your main scripts and `.env` file).

   To run the project run this python file:
   
   ```python
   python processData.py
   ```

## DensePose Repository Installation

Clone the Detectron2 repository and navigate to the DensePose project folder:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd projects/DensePose
```
## Create and Activate Conda Environment

Create a new Conda environment with Python 3.8 and activate it:

```bash
conda create -n densepose python=3.8 -y
conda activate densepose
```

## Install Dependencies

## Install the required Python dependencies:

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install torch torchvision torchaudio av
```
## Download Pretrained Model

```bash
Download the pretrained DensePose model file [`densepose_rcnn_R_50_FPN_s1x.pkl`](LINK) from Google Drive and place it in the following directory:
```

## Fix Black Background Issue

To fix the black background problem in the output images, edit the file `apply_net.py` located at:

Find and replace these two lines:


```python
image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
height, width = entry["image"].shape[:2]
image = np.zeros((height, width, 3), dtype=np.uint8)
height, width = entry["image"].shape[:2]
image = np.zeros((height, width, 3), dtype=np.uint8)
```

## Run DensePose Inference

Run DensePose Inference

```bash
python apply_net.py \
  show configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  densepose_rcnn_R_50_FPN_s1x.pkl \
  input_images/person.jpeg \
  dp_segm \
  --output output/person_output.jpg \
  --opts MODEL.DEVICE cpu

```
---

## Citations & Acknowledgments

This project makes use of the following open-source libraries and research contributions:

- [**Detectron2**](https://github.com/facebookresearch/detectron2) – Facebook AI Research's platform for object detection and segmentation.
- [**HR-VITON**](https://github.com/sangyun884/HR-VITON) – High-resolution virtual try-on network.
- [**CIHP_PGN**](https://github.com/Engineering-Course/CIHP_PGN) – Pose-guided parsing network for human part segmentation.
- [**MediaPipe Holistic Landmarker**](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) – Google’s full-body landmark detection (face, hands, and body).
- [**U²-Net**](https://github.com/xuebinqin/U-2-Net) – A powerful deep network for salient object detection.

We are grateful to the authors and maintainers of these projects for their valuable contributions to the research and open-source communities.

## 🚨 License & Usage

**This repository is only for academic or research use.**
By using any part of this project, **you agree to cite our work** in any publications, presentations, or products that incorporate it.

**Commercial use is strictly prohibited.**

Please cite using the following BibTeX entry:
```bibtex
@misc{kimo18_aifashionstyler_2025,
  author       = {Kimo18 , Amr Abdrabo},
  title        = {AI Fashion Styler},
  year         = {2025},
  howpublished = {\url{https://github.com/kimo18/AI_fashion_styler}}
}




