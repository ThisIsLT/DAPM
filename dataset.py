import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.functional as F
from utils.metric import pose_to_log_depth



class DepthPoseDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None):
        """
        root_dir: Root directory of the dataset (containing train/val subdirectories)
        phase: Dataset type (train/val/test)
        """
        self.rgb_dir = os.path.join(root_dir, phase, "rgb")  # Path concatenation method from web page 5
        self.depth_dir = os.path.join(root_dir, phase, "depth")
        
        # Get all RGB filenames (automatically match files with the same name in the depth directory) [5](@ref)
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) 
                                if os.path.isfile(os.path.join(self.rgb_dir, f))])


        
        self.transform = transform

        self.pose_to_log_depth = pose_to_log_depth

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        base_name = self.rgb_files[idx]
        
        # Construct the full path (path concatenation method from web page 1)
        rgb_path = os.path.join(self.rgb_dir, base_name)
        depth_path = os.path.join(self.depth_dir, base_name)

        # Load RGB and depth images (new code)
        rgb_image = cv2.imread(rgb_path)                   # Use cv2 to read BGR format images
        depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # Load depth map in grayscale mode

        # Add file existence check (error handling idea from web page 3)
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image missing: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image missing: {depth_path}")

        # Keep the original image processing logic unchanged...
        # Extract pose parameters from the filename
        filename = self.rgb_files[idx]
        height = float(filename.split('z_')[-1].split('_')[0])  # height
        pitch = float(filename.split('pitch_')[-1].split('_')[0])  # pitch angle
        roll = float(filename.split('roll_')[-1].split('_')[0])  # roll angle
        fov = round(float(filename.split('fov_')[-1].split('_')[0].replace('.png', '')))  # field of view

        # Convert the RGB image from BGR to RGB format, and then to a PIL image
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = T.ToPILImage()(rgb_image)

        # Apply data augmentation transformations
        if self.transform:
            rgb_image = self.transform(rgb_image)
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                rgb_image = T.functional.hflip(rgb_image)
                roll = -roll  # After flipping, the roll angle is negated

        # Depth image normalization (map from 0-255 to 0-1)
        depth_image = depth_image / 255.0

        # For grayscale images, generate binary results and one-hot encoding
        depth_binary = (depth_image == 1.0).astype(np.float32)  # Grayscale 1 corresponds to 1, less than 1 corresponds to 0

        depth_quantized_16 = np.floor(depth_image * 15).astype(np.int64)  # Uniformly quantize 0-1 to 0-15 classes
        depth_quantized_16 = np.clip(depth_quantized_16, 0, 15)  # Ensure it is between 0 and 15
        depth_one_hot_16 = np.eye(16)[depth_quantized_16].astype(np.float32)  # Generate one-hot encoding
        depth_one_hot_16 = np.transpose(depth_one_hot_16, (2, 0, 1))  # Convert to [16, H, W]

        depth_quantized_64 = np.floor(depth_image * 63).astype(np.int64)  # Uniformly quantize 0-1 to 0-63 classes
        depth_quantized_64 = np.clip(depth_quantized_64, 0, 63)  # Ensure it is between 0 and 63
        depth_one_hot_64 = np.eye(64)[depth_quantized_64].astype(np.float32)  # Generate one-hot encoding
        depth_one_hot_64 = np.transpose(depth_one_hot_64, (2, 0, 1))  # Convert to [64, H, W]


        # Pose label normalization
        height = (height - 10) / 90
        pitch = (pitch + 100) / 110
        roll = (roll + 30) / 60
        fov = (fov - 30) / 90

        # For pose parameters, generate one-hot encoding
        height_quantized_16 = np.floor(height * 15).astype(np.int64)  # Uniformly quantize height to 0-15 classes
        height_quantized_16 = np.clip(height_quantized_16, 0, 15)
        pitch_quantized_16 = np.floor(pitch * 15).astype(np.int64)  # Uniformly quantize pitch angle to 0-15 classes
        pitch_quantized_16 = np.clip(pitch_quantized_16, 0, 15)
        roll_quantized_16 = np.floor(roll * 15).astype(np.int64)  # Uniformly quantize roll angle to 0-15 classes
        roll_quantized_16 = np.clip(roll_quantized_16, 0, 15)
        fov_quantized_16 = np.floor(fov * 15).astype(np.int64)  # Uniformly quantize field of view to 0-15 classes
        fov_quantized_16 = np.clip(fov_quantized_16, 0, 15)

        height_one_hot_16 = np.eye(16)[height_quantized_16].astype(np.float32)  # Generate one-hot encoding for height
        pitch_one_hot_16 = np.eye(16)[pitch_quantized_16].astype(np.float32)  # Generate one-hot encoding for pitch angle
        roll_one_hot_16 = np.eye(16)[roll_quantized_16].astype(np.float32)  # Generate one-hot encoding for roll angle
        fov_one_hot_16 = np.eye(16)[fov_quantized_16].astype(np.float32)  # Generate one-hot encoding for field of view

        pose_one_hot_16 = np.stack([height_one_hot_16, pitch_one_hot_16, roll_one_hot_16, fov_one_hot_16], axis=0)  # Convert to [4, 16]



        height_quantized_64 = np.floor(height * 63).astype(np.int64)  # Uniformly quantize height to 0-63 classes
        height_quantized_64 = np.clip(height_quantized_64, 0, 63)
        pitch_quantized_64 = np.floor(pitch * 63).astype(np.int64)  # Uniformly quantize pitch angle to 0-63 classes
        pitch_quantized_64 = np.clip(pitch_quantized_64, 0, 63)
        roll_quantized_64 = np.floor(roll * 63).astype(np.int64)  # Uniformly quantize roll angle to 0-63 classes
        roll_quantized_64 = np.clip(roll_quantized_64, 0, 63)
        fov_quantized_64 = np.floor(fov * 63).astype(np.int64)  # Uniformly quantize field of view to 0-63 classes
        fov_quantized_64 = np.clip(fov_quantized_64, 0, 63)

        height_one_hot_64 = np.eye(64)[height_quantized_64].astype(np.float32)  # Generate one-hot encoding for height
        pitch_one_hot_64 = np.eye(64)[pitch_quantized_64].astype(np.float32)  # Generate one-hot encoding for pitch angle
        roll_one_hot_64 = np.eye(64)[roll_quantized_64].astype(np.float32)  # Generate one-hot encoding for roll angle
        fov_one_hot_64 = np.eye(64)[fov_quantized_64].astype(np.float32)  # Generate one-hot encoding for field of view

        pose_one_hot_64 = np.stack([height_one_hot_64, pitch_one_hot_64, roll_one_hot_64, fov_one_hot_64], axis=0)  # Convert to [4, 64]



        # Convert the depth image to a tensor (no need to call T.ToTensor() again)
        depth_image = torch.tensor(depth_image, dtype=torch.float32).unsqueeze(0)
        depth_binary = torch.tensor(depth_binary, dtype=torch.float32).unsqueeze(0)
        depth_one_hot_16 = torch.tensor(depth_one_hot_16, dtype=torch.float32)
        depth_one_hot_64 = torch.tensor(depth_one_hot_64, dtype=torch.float32)

        # Convert pose parameters to a tensor
        pose_tensor = torch.tensor([height, pitch, roll, fov], dtype=torch.float32)
        pose_one_hot_16 = torch.tensor(pose_one_hot_16, dtype=torch.float32)
        pose_one_hot_64 = torch.tensor(pose_one_hot_64, dtype=torch.float32)

        # print(pose_tensor.shape)
        p2d_label = self.pose_to_log_depth(pose_tensor.unsqueeze(0))

        # Return RGB image, depth image, binary result, depth one-hot encoding, pose parameters, and pose one-hot encoding
        return rgb_image, depth_image, depth_binary, depth_one_hot_16, depth_one_hot_64, pose_tensor, pose_one_hot_16, pose_one_hot_64, p2d_label