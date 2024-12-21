<div id="top" align="center">

# DPF-Net 
**Physical Imaging Model Embedded Data-Driven Underwater Image Restoration Network**

<img src="./assets/results.png" width="90%" alt="teaser" align=center />

<div id="top" align="left">
## ⚙️ Setup

We ran our experiments with PyTorch 2.3.0, CUDA 12.1, Python 3.9.19 and Ubuntu 18.04.

Relative depth estimates for this project are based on Depth-Anything-V2, which you can find [here](https://github.com/DepthAnything/Depth-Anything-V2).

## 💾 Data Preparation
**Dataset**

We mainly used the UIEB dataset. You can download the UIEB and  UIEB-Challenging dataset from [here](https://opendatalab.com/OpenDataLab/UIEB) and pre-convert the images to.jpg format. 

As for Degraded Parameters Estimation Module (DPEM), you can download the NYU-Depth-V2 dataset from [here](https://opendatalab.com/OpenDataLab/NYUv2). The absolute depth scale of each image extracted is saved in the file ./DPEM/depth_scale.txt.

## 📦 Models

You can download the model weights we provided [here](https://drive.google.com/drive/folders/1rZe1U5Sq0IrEFXv3vV6KUIIkVb5Qa4ON?usp=sharing), including **DPEM**(trained on the synthetic data in the first stage), **DPEM_finetune** (fine-tuned at a low learning rate in the second stage), **DPF-Net** (for image enhancement) and **Depth-Anything-V2** (for generating depth maps when data is loaded, you can also substitute other MDE models if you like)
