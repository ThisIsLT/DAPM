# DAPM：UAV Monocular Depth Estimation from Any Height, Pitch, Roll and FOV

## Abstract

Monocular depth estimation is a fundamental prerequisite for 3D reconstruction and autonomous navigation in Unmanned Aerial Vehicles (UAVs). In practical deployments, UAVs operate under highly dynamic camera poses characterized by continuous variations in height, pitch, roll, and field of view (FOV). Existing monocular depth estimation methods frequently fail to generalize across such diverse perspectives and the expansive scale of depth distributions inherent in aerial scenes. To address these challenges, we establish a quantitative representation of UAV viewing angles through rigorous theoretical analysis, deriving the geometric correspondence between viewing angles and view distances using the ground plane as a reference for observation. Building upon this, we propose Depth Estimation for Any Perspectives Model (DAPM), representing the first monocular framework specifically designed for UAV aerial imagery to jointly estimate camera pose and depth under continuously varying viewpoints. Specifically, we introduce an Ideal Ground Depth (IGD) module that leverages the derived geometric relationships between UAV perspectives and view distances to implement dense camera-pose supervision and enhance depth features. And we further develop a coarse-to-fine Progressive Quantization Bins (PQB) module. By incorporating progressive supervision and hierarchical quantization bins, the PQB module enables robust estimation in complex UAV aerial imagery. To evaluate the proposed framework, we present the UAV Any Perspectives Depth (UAPD) dataset, featuring comprehensive and continuous distributions of pose parameters. Experimental results on UAPD demonstrate that DAPM achieves state-of-the-art performance across both depth and camera-pose estimation metrics.


## Setup

### Prerequisites
* Linux or Windows
* Python 3.8
* NVIDIA GPU + CUDA 11.8

### Quick Start
Run the following commands to set up the environment:

```bash
# 1. Create and activate environment
conda create -n DAPM python=3.8 -y
conda activate DAPM

# 2. Install binary libs (required for video/image processing)
conda install opencv=4.5.4 -c defaults -y

# 3. Install python libs
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Train the model using the following command:

```
python train.py
```

Run model inference using the following command:

```
python test.py
```

## UAPD dataset

**The dataset is available at the following link:**
https://pan.baidu.com/s/1wP8FzE-KBbYsO_diMaxvdQ?pwd=k37s

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/UAPD.jpg" width="800" height="500">
</center>

By selecting simulation scenes, setting the time, and generating random pedestrians and vehicles, a realistic simulation environment is constructed. Subsequently, images and their corresponding depth maps are acquired by randomly sampling camera positions and poses. In addition, we provide the data proportions of different scenes and the distribution of viewing perspectives within the UAPD dataset.


<!-- <center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/vis_ms.jpg" width="800" height="500">
</center>

**Overall Model Structure.** DAPM first extracts features using a shared encoder and then applies separate decoders for pose and depth to obtain initial estimates. It subsequently refines these predictions using the Progressive Quantization Bins and Depth Upper-Bound modules. Compared to previous methods that utilize pose estimation to assist depth prediction, DAPM demonstrates clear advantages in UAV imagery through its innovative design.

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/pqb.jpg" width="800" height="500">
</center>

**Progressive Quantification Bins Module.** This module gradually improves the estimation accuracy of camera pose and depth by continuously increasing the number of bins for classification estimation. Each bin block obtains the corresponding classification result by concatenating the input features and performing convolution processing. Finally, the depth head and pose head fuse all the preceding features to give the final estimation result. -->


## Visual Comparison

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/vis_0802_2_horizontal.jpg" width="800" height="500">
</center>

**Qualitative comparison of different depth estimation methods and the DAPM on the UAPD dataset.** The results reveal that DAPM consistently produces more accurate depth estimates across a wide range of viewing angles, which clearly demonstrates its significant advantages over existing methods.

