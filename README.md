1. Clone DensePose Repo

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd projects/DensePose



2. Create and Activate Conda Environment (recommended)

conda create -n densepose python=3.8 -y
conda activate densepose

3. Install Dependancies

pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install torch torchvision torchaudio av

4. Download from google drive this file densepose_rcnn_R_50_FPN_s1x.pkl {LINK}
and put it in /detectron2/projects/DensePose/densepose_rcnn_R_50_FPN_s1x.pkl

5. Black Background

in the apply_net.py {/detectron2/projects/DensePose/apply_net.py}
change the 2 lines 
{
image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
}
to these lines
{
height, width = entry["image"].shape[:2]
image = np.zeros((height, width, 3), dtype=np.uint8)
}

6. Run this in terminal 

python apply_net.py \
  show configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  densepose_rcnn_R_50_FPN_s1x.pkl \
  input_images/person.jpeg \
  dp_segm \
  --output output/person_output.jpg \
  --opts MODEL.DEVICE cpu