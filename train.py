import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import time
from model.dapm import EfficientDepthPoseNet
from utils.metric import calculate_pose_errors, compute_depth_metrics
from dataset import DepthPoseDataset
from utils.log import log_message, log_config, log_errors, log_loss, log_pose_errors
from utils.loss import scale_invariant_loss, gard_weight_loss
 
batchsize = 32
numworkers = 8
epochs = 24
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
# Define log folder path and model save folder path
project_name = time.strftime('%Y%m%d_%H%M%S') + '_v1'
log_dir = f"/root/pose_depth/log/{project_name}"
checkpoint_dir = f"/root/pose_depth/checkpoint/{project_name}"



# Create log folder and checkpoint folder if they don't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

log_file_name = f"{log_dir}/{project_name}.txt"



# Data augmentation and transformation operations
transform = T.Compose([
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    T.ToTensor(),  # Convert PIL image or NumPy array to Tensor
    T.RandomApply([T.Lambda(lambda x: x + torch.randn_like(x) * 0.05)], p=0.5)
])

transform_val = T.Compose([
    T.ToTensor()  # Only convert image to Tensor
])


dataset_root = "UAPD"  # Root directory path
train_dataset = DepthPoseDataset(root_dir=dataset_root, phase="train", transform=transform)
val_dataset = DepthPoseDataset(root_dir=dataset_root, phase="val", transform=transform_val)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=numworkers, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=numworkers, drop_last=True)



if __name__ == '__main__':

    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_path = 'xx.pth'
    start_epoch = 0
    best_val_loss = float('inf')

    checkpoint_queue = []  # Store the last three checkpoint paths

    # Device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = EfficientDepthPoseNet().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}")
        model = torch.nn.DataParallel(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)

    # train_dataloader
    total_iters = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_iters,
        pct_start=0.30,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=100
    )

    # Define loss function
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()


    # If resume checkpoint exists, load it
    if os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)

    # Log configuration
    log_config(model, optimizer, scheduler)


    # Start training from start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        for step, (rgb_images, depth_images, depth_binary,
                   depth_one_hot_16, depth_one_hot_64,
                   pose_labels, pose_one_hot_16, pose_one_hot_64, p2d_label) in enumerate(train_dataloader):

            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)
            depth_binary = depth_binary.to(device)
            depth_one_hot_16 = depth_one_hot_16.to(device)
            depth_one_hot_64 = depth_one_hot_64.to(device)
            pose_one_hot_16 = pose_one_hot_16.to(device)
            pose_one_hot_64 = pose_one_hot_64.to(device)
            pose_labels = pose_labels.to(device)
            p2d_label = p2d_label.to(device)

            optimizer.zero_grad()
            (pred_init_depth, pred_depth, pred_init_pose, pred_pose,
            pred_depth_binary,
            pred_depth_q16, pred_depth_q64,
            pred_pose_q16, pred_pose_q64, pred_p2d) = model(rgb_images)


            loss_bce = criterion_bce(pred_depth_binary, depth_binary)
            loss_dq16 = criterion_ce(pred_depth_q16, depth_one_hot_16.argmax(dim=1))
            loss_dq64 = criterion_ce(pred_depth_q64, depth_one_hot_64.argmax(dim=1))
            loss_init_depth = scale_invariant_loss(pred_init_depth.squeeze(1), depth_images.squeeze(1), λ=0.85, a=10.0)
            loss_depth = scale_invariant_loss(pred_depth.squeeze(1), depth_images.squeeze(1), λ=0.85, a=10.0)
            depth_grad_loss = gard_weight_loss(predicted_depth, depth_images)

            loss_pq16 = criterion_ce(pred_pose_q16.permute(0,2,1), pose_one_hot_16.argmax(dim=2))
            loss_pq64 = criterion_ce(pred_pose_q64.permute(0,2,1), pose_one_hot_64.argmax(dim=2))
            loss_init_pose = criterion_mse(pred_init_pose, pose_labels)
            loss_pose = criterion_mse(pred_pose, pose_labels)

            loss_p2d = criterion_mse(pred_p2d.squeeze(), p2d_label.squeeze())


            loss_init = loss_init_depth + loss_init_pose
            loss_dq = loss_bce + loss_dq16 + loss_dq64 
            loss_pq = loss_pq16 + loss_pq64

            total_loss = (
                loss_depth + 10 * loss_pose + loss_init +
                loss_dq + loss_pq + 10 * loss_p2d + 10 * depth_grad_loss
            )


            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 5 == 0:
                h_err, p_err, r_err, f_err, d_err = calculate_pose_errors(
                    pred_pose, pose_labels, pred_depth, depth_images)
                log_errors(epoch, step, "Train",
                           d_err.mean().item(), h_err.mean().item(),
                           p_err.mean().item(), r_err.mean().item(), f_err.mean().item())
                log_loss(epoch, step, total_loss,
                         loss_init_depth, loss_depth, loss_init_pose, loss_pose,
                         loss_bce, loss_dq16, loss_dq64, depth_grad_loss, loss_pq16, loss_pq64, loss_p2d)


        # Validation phase
        all_pred = []
        all_gt   = []
        all_h_errs, all_p_errs, all_r_errs, all_f_errs, all_d_errs = [], [], [], [], []
        model.eval()
        val_loss = 0.0
        total_height_error, total_pitch_error, total_roll_error, total_fov_error, total_depth_error = 0, 0, 0, 0, 0
        for step, (rgb_images, depth_images, depth_binary, depth_one_hot_16, depth_one_hot_64, pose_labels, pose_one_hot_16, pose_one_hot_64, p2d_label) in enumerate(val_dataloader):

            # Move data to device
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)
            depth_binary = depth_binary.to(device)
            depth_one_hot_16 = depth_one_hot_16.to(device)
            depth_one_hot_64 = depth_one_hot_64.to(device)
            pose_one_hot_16 = pose_one_hot_16.to(device)
            pose_one_hot_64 = pose_one_hot_64.to(device)
            pose_labels = pose_labels.to(device)
            p2d_label = p2d_label.to(device)

            with torch.no_grad():
                # Forward propagation
                (pred_init_depth, pred_depth, pred_init_pose, pred_pose,
                pred_depth_binary,
                pred_depth_q16, pred_depth_q64,
                pred_pose_q16, pred_pose_q64, pred_p2d) = model(rgb_images)

            all_pred.append(pred_depth.squeeze(1).cpu())
            all_gt.append(depth_images.squeeze(1).cpu())

            # Calculate loss
            loss_bce = criterion_bce(pred_depth_binary, depth_binary)
            loss_dq16 = criterion_ce(pred_depth_q16, depth_one_hot_16.argmax(dim=1))
            loss_dq64 = criterion_ce(pred_depth_q64, depth_one_hot_64.argmax(dim=1))
            loss_init_depth = scale_invariant_loss(pred_init_depth.squeeze(1), depth_images.squeeze(1), λ=0.85, a=10.0)
            loss_depth = scale_invariant_loss(pred_depth.squeeze(1), depth_images.squeeze(1), λ=0.85, a=10.0)
            depth_grad_loss = gard_weight_loss(predicted_depth, depth_images)

            loss_pq16 = criterion_ce(pred_pose_q16.permute(0,2,1), pose_one_hot_16.argmax(dim=2))
            loss_pq64 = criterion_ce(pred_pose_q64.permute(0,2,1), pose_one_hot_64.argmax(dim=2))
            loss_init_pose = criterion_mse(pred_init_pose, pose_labels)
            loss_pose = criterion_mse(pred_pose, pose_labels)

            loss_p2d = criterion_mse(pred_p2d.squeeze(), p2d_label.squeeze())

            loss_init = loss_init_depth + loss_init_pose
            loss_dq = loss_bce + loss_dq16 + loss_dq64 
            loss_pq = loss_pq16 + loss_pq64

            total_loss = (
                loss_depth + 10 * loss_pose + loss_init +
                loss_dq + loss_pq + 10 * loss_p2d + 10 * depth_grad_loss
            )

            val_loss += total_loss.item()

            # Calculate and accumulate errors
            height_error, pitch_error, roll_error, fov_error, depth_error = calculate_pose_errors(pred_pose, pose_labels, pred_depth, depth_images)
            total_height_error += height_error.mean().item()
            total_pitch_error += pitch_error.mean().item()
            total_roll_error += roll_error.mean().item()
            total_fov_error += fov_error.mean().item()
            total_depth_error += depth_error.mean().item()

            all_h_errs.append(height_error.cpu())
            all_p_errs.append(pitch_error.cpu())
            all_r_errs.append(roll_error.cpu())
            all_f_errs.append(fov_error.cpu())
            all_d_errs.append(depth_error.cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_gt   = torch.cat(all_gt,   dim=0)
        all_h_errs = torch.cat(all_h_errs, dim=0)
        all_p_errs = torch.cat(all_p_errs, dim=0)
        all_r_errs = torch.cat(all_r_errs, dim=0)
        all_f_errs = torch.cat(all_f_errs, dim=0)
        all_d_errs = torch.cat(all_d_errs, dim=0)

        # Compute all depth metrics
        val_metrics = compute_depth_metrics(all_pred, all_gt)

        # Log validation errors and loss
        avg_val_loss = val_loss / len(val_dataloader)
        avg_height_error = total_height_error / len(val_dataloader)
        avg_pitch_error = total_pitch_error / len(val_dataloader)
        avg_roll_error = total_roll_error / len(val_dataloader)
        avg_fov_error = total_fov_error / len(val_dataloader)
        avg_depth_error = total_depth_error / len(val_dataloader)

        log_message(f" ")
        log_message(f"Validation Depth Metrics:")
        for name, value in val_metrics.items():
            log_message(f"  {name}: {value:.4f}")
        log_pose_errors(all_h_errs, all_p_errs, all_r_errs, all_f_errs, all_d_errs,
                epoch, epochs, tag="Validation")
        log_errors(epoch, 0, "Validation", avg_depth_error, avg_height_error, avg_pitch_error, avg_roll_error, avg_fov_error)
        log_loss(epoch, total_loss,
                         loss_init_depth, loss_depth, loss_init_pose, loss_pose,
                         loss_bce, loss_dq16, loss_dq64, depth_grad_loss, loss_pq16, loss_pq64, loss_p2d)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = f"{checkpoint_dir}/best_model.pth"
            # Handle multi-GPU parallel training
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics
            }, best_checkpoint_path)
            log_message(f"New best model saved at {best_checkpoint_path} with val_loss: {avg_val_loss:.4f}")

        # Save current epoch checkpoint
        current_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'val_metrics': val_metrics
        }

        current_checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}_checkpoint.pth"
        torch.save(current_checkpoint, current_checkpoint_path)
        checkpoint_queue.append(current_checkpoint_path)

        # Keep at most 3 checkpoints
        while len(checkpoint_queue) > 3:
            old_checkpoint = checkpoint_queue.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                log_message(f"Removed old checkpoint: {old_checkpoint}")

        log_message(f"Current checkpoints preserved: {[os.path.basename(p) for p in checkpoint_queue[-3:]]}")