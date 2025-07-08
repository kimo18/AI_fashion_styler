# Fashion AI Setup and Usage Guide

This guide explains how to clone, set up, and run the project from Facebook Research.

---

## Table of Contents


- [Create and Activate Conda Environment](#create-and-activate-conda-environment)
- [Densepose Repository Installation](#densepose-repository-installation)
- [Install Dependencies](#install-dependencies)
- [Download Pretrained Model](#download-pretrained-model)
- [Fix Black Background Issue](#fix-black-background-issue)
- [Run DensePose Inference](#run-densepose-inference)
- [Setup Installation](#setup-installation)


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



