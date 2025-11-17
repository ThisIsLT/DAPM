# DAPM：UAV Monocular Depth Estimation from Any Height, Pitch, Roll and FOV


## UAPD dataset

Train the model using the following command:

```
python train.py
```

Run model inference using the following command:

```
python test.py
```



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

