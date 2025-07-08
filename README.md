# DensePose Setup and Usage Guide

This guide explains how to clone, set up, and run the DensePose project from Facebook Research.

---

## Table of Contents

- [Clone Repository](#clone-repository)
- [Create and Activate Conda Environment](#create-and-activate-conda-environment)
- [Install Dependencies](#install-dependencies)
- [Download Pretrained Model](#download-pretrained-model)
- [Fix Black Background Issue](#fix-black-background-issue)
- [Run DensePose Inference](#run-densepose-inference)

---

## Clone Repository

Clone the Detectron2 repository and navigate to the DensePose project folder:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd projects/DensePose
