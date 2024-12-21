<div id="top" align="center">

# DPF-Net 
**Physical Imaging Model Embedded Data-Driven Underwater Image Restoration Network**

<img src="./assets/results.png" width="90%" alt="teaser" align=center />

## ⚙️ Setup

We ran our experiments with PyTorch 2.3.0, CUDA 12.1, Python 3.9.19 and Ubuntu 18.04.

Relative depth estimates for this project are based on Depth-Anything-V2, which you can find [here](https://github.com/DepthAnything/Depth-Anything-V2).

## 💾 Data Preparation
**Dataset**

We mainly used the UIEB dataset. You can download the UIEB and  UIEB-Challenging dataset from [here](https://opendatalab.com/OpenDataLab/UIEB) and pre-convert the images to.jpg format. 

As for Degraded Parameters Estimation Module (DPEM), you can download the NYU-Depth-V2 dataset from [here](https://opendatalab.com/OpenDataLab/NYUv2). The absolute depth scale of each image extracted is saved in the file ./DPEM/depth_scale.txt.

## 📦 Models

You can download the model weights we provided [here](https://drive.google.com/drive/folders/1rZe1U5Sq0IrEFXv3vV6KUIIkVb5Qa4ON?usp=sharing), including **DPEM**(trained on the synthetic data in the first stage), **DPEM_finetune** (fine-tuned at a low learning rate in the second stage), **DPF-Net** (for image enhancement) and **Depth-Anything-V2** (for generating depth maps when data is loaded, you can also substitute other MDE models if you like)


## 📊 Test and Evaluation
**Test**

You can predict disparity for a single image with:

    python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image

**Evaluation**

If you want to evaluate the model on the test set defined by `OUC_split`, first prepare the ground truth depth maps by running:

    python export_gt_depth.py

Then evaluate the model by running:

    python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/FLSea_data/ --model lite-mono

If you want to test generalization on the FLSea-stereo dataset, please add flag `--eval_stereo`.

## 🕒Training
The code of training will be available after the paper is received.
#### start training
    python train.py --data_path path/to/your/data --model_name mytrain --num_epochs 30 --batch_size 12

#### tensorboard visualization
    tensorboard --log_dir ./tmp/mytrain

## 💕Thanks
Our code is based on [Monodepth2](https://github.com/nianticlabs/monodepth2), [Lite-Mono](https://github.com/noahzn/Lite-Mono) and [Sea-thru](https://github.com/hainh/sea-thru). You can refer to their README files and source code for more implementation details. 

## 🖇️Citation

    None