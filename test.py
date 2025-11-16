import os
import re
import cv2
import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from model.dapm import EfficientDepthPoseNet, pose_to_log_depth
from torch.utils.data.dataloader import default_collate

# Configuration
input_rgb_dir = '/UAPD/test/rgb'
input_depth_dir = '/UAPD/test/depth'
checkpoint_path = '/UAPD/checkpoint/20250726_210211/best_model.pth'
output_dir = '/result'
result_file_path = os.path.join(output_dir, 'predictions.txt')
metrics_file_path = os.path.join(output_dir, 'metrics_summary.txt')



# Output image subdirectories
subdirs = {
    'depth_final': 'depth_final',
    'binary_mask': 'binary_mask',
    'p2d': 'p2d_pred',
    'gt_p2d': 'p2d_gt'
}
for d in subdirs.values():
    os.makedirs(os.path.join(output_dir, d), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'depth_quant_16'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'depth_quant_64'), exist_ok=True)

# Model Loading
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientDepthPoseNet()
ckpt = torch.load(checkpoint_path, map_location='cpu')
if 'module.' in list(ckpt['model_state_dict'].keys())[0]:
    model = torch.nn.DataParallel(model)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

# Dataset
class DepthPoseDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.filenames = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.rgb_dir, fname)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(self.depth_dir, fname), cv2.IMREAD_GRAYSCALE)
        return self.transform(rgb), torch.tensor(depth / 255.0 + 1e-6).unsqueeze(0), fname

def collate_keep_fname(batch):
    images, depths, fnames = zip(*batch)
    return default_collate(images), default_collate(depths), fnames

# Image Saving
def save_map(tensor, subdir, fname, suffix, color=False):
    arr = tensor.squeeze().detach().cpu().float().numpy()
    

    if arr.ndim == 3:
        arr = np.argmax(arr, axis=0)

    
    if 'quant16' in subdir:
        out = (arr / 15.0 * 255.0).astype(np.uint8)
    elif 'quant64' in subdir:
        out = (arr / 63.0 * 255.0).astype(np.uint8)
    elif 'binary_mask' in subdir and not color:
        arr = np.clip(arr, 0, 1) 
        out = (arr > 0.5).astype(np.uint8) * 255
    else:
        arr = np.clip(arr, 0, 1)
        out = (arr * 255).astype(np.uint8)
    


    save_path = os.path.join(output_dir, subdir, fname.replace('.png', f'_{suffix}.png'))

    if color:
        out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        
    cv2.imwrite(save_path, out)



def save_tensor_as_image(tensor, path, quant_max=None, color=False):
    """
    tensor: torch.Tensor, shape [B,C,H,W] or [C,H,W] or [1,H,W]
    path: save path
    quant_max: int, 如果是量化图，需要归一化到 [0,255]
    color: bool, 是否应用伪彩色
    """
    arr = tensor.detach().cpu().float()
    if arr.ndim == 4:
        arr = arr[0]  # 去 batch
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # [H,W]
    elif arr.ndim == 3 and quant_max is not None:
        arr = torch.argmax(arr, dim=0).float()
        arr = arr / quant_max * 255.0
    arr = np.clip(arr.numpy(), 0, 255).astype(np.uint8)
    if color:
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    cv2.imwrite(path, arr)



# Evaluation Functions
def calculate_pose_errors(pred, gt, pred_depth, gt_depth):
    height_errs = torch.abs(pred[:, 0] - gt[:, 0]) * 90
    pitch_errs = torch.abs(pred[:, 1] - gt[:, 1]) * 110
    roll_errs = torch.abs(pred[:, 2] - gt[:, 2]) * 60
    fov_errs = torch.abs(pred[:, 3] - gt[:, 3]) * 90
    depth_errs = torch.abs(pred_depth - gt_depth) * 255
    return height_errs, pitch_errs, roll_errs, fov_errs, depth_errs

def compute_depth_metrics(pred, gt):
    pred *= 255
    gt *= 255
    pred = torch.clamp(pred, min=1e-3)
    gt = torch.clamp(gt, min=1e-3)
    ratio = torch.max(pred / gt, gt / pred)
    δ1 = (ratio < 1.25).float().mean()
    δ2 = (ratio < 1.25 ** 2).float().mean()
    δ3 = (ratio < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(pred) - torch.log(gt)) ** 2))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(gt)))
    d = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(d ** 2) - torch.mean(d) ** 2) * 100.0
    irmse = torch.sqrt(torch.mean((1.0 / pred - 1.0 / gt) ** 2))
    return {
        'δ1': δ1.item(), 'δ2': δ2.item(), 'δ3': δ3.item(),
        'AbsRel': abs_rel.item(), 'SqRel': sq_rel.item(),
        'RMSE': rmse.item(), 'RMSElog': rmse_log.item(),
        'log10': log10.item(), 'SILog': silog.item(), 'iRMSE': irmse.item(),
    }

def compute_auc(errs, thresholds=[1.0, 5.0, 10.0]):
    aucs = {}
    recalls = [(errs <= t).float().mean().item() for t in thresholds]
    xs = [0.0] + thresholds
    ys = [0.0] + recalls
    for i, T in enumerate(thresholds, 1):
        area = sum(0.5 * (ys[j] + ys[j+1]) * (xs[j+1] - xs[j]) for j in range(i))
        aucs[f"AUC@{int(T)}°"] = area / T
    return aucs

# Inference
dataloader = DataLoader(DepthPoseDataset(input_rgb_dir, input_depth_dir), batch_size=1, shuffle=False, collate_fn=collate_keep_fname)

all_results, all_h, all_pitch, all_roll, all_fov, all_depth = [], [], [], [], [], []

start_time = time.time()
total_frames = len(dataloader)

with torch.no_grad():
    for step, (rgb, depth, fnames) in enumerate(dataloader):
        rgb, depth = rgb.to(device), depth.to(device)
        fname = fnames[0]

        _, depth_final, _, pose_pred, binary_mask, depth_quant_16, depth_quant_64, _, _, p2d_pred = model(rgb)

        # Parse GT pose from filename
        m = re.search(r"_z_([-+]?\d+\.\d+)_pitch_([-+]?\d+\.\d+)_yaw_([-+]?\d+\.\d+)_roll_([-+]?\d+\.\d+)_fov_([-+]?\d+\.\d+)", fname)
        if not m:
            raise ValueError(f"Could not parse GT info from: {fname}")
        z_gt, pitch_gt, yaw_gt, roll_gt, fov_gt = map(float, m.groups())
        pose_gt = torch.tensor([[
            (z_gt - 10) / 90,
            (pitch_gt + 100) / 110,
            (roll_gt + 30) / 60,
            (fov_gt - 30) / 90
        ]], device=device)

        # Error calculation
        h_err, pitch_err, roll_err, fov_err, depth_err = calculate_pose_errors(pose_pred, pose_gt, depth_final, depth)
        all_h.append(h_err.cpu())
        all_pitch.append(pitch_err.cpu())
        all_roll.append(roll_err.cpu())
        all_fov.append(fov_err.cpu())
        all_depth.append(depth_err.mean().cpu())

        # Save images
        outputs = {
            'depth_final': depth_final,
            'binary_mask': binary_mask,
            'p2d': p2d_pred,
            'gt_p2d': pose_to_log_depth(pose_gt),
        }
        for key, tensor in outputs.items():
            save_map(tensor, subdirs[key], fname, key)

        save_tensor_as_image(depth_quant_16, os.path.join(output_dir,'depth_quant_16', fname), quant_max=15)
        save_tensor_as_image(depth_quant_64, os.path.join(output_dir,'depth_quant_64', fname), quant_max=63)

        # Write result line
        result_line = (
            f"{fname} - Pred(H: {(pose_pred[0,0]*90+10):.2f}, P: {(pose_pred[0,1]*110-100):.2f}, R: {(pose_pred[0,2]*60-30):.2f}, FOV: {(pose_pred[0,3]*90+30):.2f})"
            f" | Errors → H: {h_err.item():.2f}, Pitch: {pitch_err.item():.2f}, Roll: {roll_err.item():.2f}, FOV: {fov_err.item():.2f}, Depth MAE: {depth_err.mean().item():.2f}\n"
        )
        all_results.append(result_line)
        print(f"[{step+1}/{len(dataloader)}] {result_line.strip()}")

end_time = time.time()
total_time = end_time - start_time
fps = total_frames / total_time
print(f"\nInference finished. Processed {total_frames} images in {total_time:.2f} seconds.")
print(f"FPS: {fps:.2f}")

# Write predictions.txt
with open(result_file_path, 'w') as f:
    f.writelines(all_results)

# Aggregate and write metrics_summary.txt
all_h = torch.cat(all_h)
all_pitch = torch.cat(all_pitch)
all_roll = torch.cat(all_roll)
all_fov = torch.cat(all_fov)
all_depth = torch.stack(all_depth)

pose_metrics = {
    'depth': {'MAE': all_depth.mean().item(), 'Median': all_depth.median().item()},
    'height': {'MAE': all_h.mean().item(), 'Median': all_h.median().item(), **compute_auc(all_h)},
    'pitch':  {'MAE': all_pitch.mean().item(), 'Median': all_pitch.median().item(), **compute_auc(all_pitch)},
    'roll':   {'MAE': all_roll.mean().item(), 'Median': all_roll.median().item(), **compute_auc(all_roll)},
    'fov':    {'MAE': all_fov.mean().item(), 'Median': all_fov.median().item(), **compute_auc(all_fov)}
}
depth_metrics = compute_depth_metrics(depth_final.squeeze(), depth.squeeze())

with open(metrics_file_path, 'w') as f:
    f.write(f"FPS: {fps:.2f}\n\n")
    f.write("Depth Metrics (from last sample):\n")
    for k, v in depth_metrics.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\nPose Metrics (aggregated):\n")
    f.write(f"  Depth → MAE: {pose_metrics['depth']['MAE']:.4f}, Median: {pose_metrics['depth']['Median']:.4f}\n")
    for key in ['height', 'pitch', 'roll', 'fov']:
        m = pose_metrics[key]
        aucs = ", ".join([f"{k}: {v:.2f}" for k, v in m.items() if k.startswith('AUC')])
        f.write(f"  {key.capitalize():<6} → MAE: {m['MAE']:.4f}, Median: {m['Median']:.4f}, {aucs}\n")

print(f"\nMetrics summary saved to: {metrics_file_path}")
















