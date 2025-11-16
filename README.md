# DAPM：UAV Monocular Depth Estimation from Any Height, Pitch, Roll and FOV

**The dataset is available at the following link:**
https://pan.baidu.com/s/1wP8FzE-KBbYsO_diMaxvdQ?pwd=k37s

Depth estimation from imagery is of vital importance for enabling robust 3D reconstruction and navigation in UAVs. In practice, UAVs experience variations in camera pose, including height, pitch, roll, and field of view (FOV). Existing depth estimation methods often struggle to generalize across such diverse and continuous distribution perspectives, thus restricting their practical deployment. To address the above issue, we propose Depth Estimation for Any Perspectives Model (DAPM), a coarse-to-fine monocular framework with progressive supervision and hierarchically refined quantization bins, enabling robust depth estimation under diverse UAV poses. DAPM is the first method for monocular UAV imagery that jointly estimates camera pose and depth under continuously varying viewpoints. We further propose a Depth Upper-Bound Refinement module that leverages estimated pose parameters to perform geometric projection, deriving pixel-wise ground plane distances as geometry-aware priors for enhanced depth estimation. In addition, we introduce a Depth Gradient-Weighted Loss (DGWL), which combines the loss of predicted depth gradients with a weighting of the predicted depth error by the ground-truth depth gradients, thereby improving the model's sensitivity to regions with significant depth variations. To validate the effectiveness of the model, we present the UAV Any Perspectives Depth (UAPD) dataset, which we constructed with continuously distributed pose parameters including height, pitch, roll, and FOV. Experimental results on UAPD demonstrate that DAPM achieves state-of-the-art performance across multiple metrics in depth and camera pose estimation.

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/UAPD.jpg" width="800" height="500">
</center>

**UAPD dataset.** By selecting simulation scenes, setting the time, and generating random pedestrians and vehicles, a realistic simulation environment is constructed. Subsequently, images and their corresponding depth maps are acquired by randomly sampling camera positions and poses. In addition, we provide the data proportions of different scenes and the distribution of viewing perspectives within the UAPD dataset.

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/vis_ms.jpg" width="800" height="500">
</center>

**Overall Model Structure.** DAPM first extracts features using a shared encoder and then applies separate decoders for pose and depth to obtain initial estimates. It subsequently refines these predictions using the Progressive Quantization Bins and Depth Upper-Bound modules. Compared to previous methods that utilize pose estimation to assist depth prediction, DAPM demonstrates clear advantages in UAV imagery through its innovative design.

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/pqb.jpg" width="800" height="500">
</center>

**Progressive Quantification Bins Module.** This module gradually improves the estimation accuracy of camera pose and depth by continuously increasing the number of bins for classification estimation. Each bin block obtains the corresponding classification result by concatenating the input features and performing convolution processing. Finally, the depth head and pose head fuse all the preceding features to give the final estimation result.

<center>
<img src="https://github.com/ThisIsLT/DAPM/blob/main/fig/vis_0802_2_horizontal.jpg" width="800" height="500">
</center>

**Qualitative comparison of different depth estimation methods and the DAPM on the UAPD dataset.** The results reveal that DAPM consistently produces more accurate depth estimates across a wide range of viewing angles, which clearly demonstrates its significant advantages over existing methods.

