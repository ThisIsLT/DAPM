import torch
from train import log_file_name, epochs, train_dataloader, batchsize
from utils.metric import compute_auc_per_threshold

# Write training information to the log file
def log_message(message):
    with open(log_file_name, 'a') as log_file:
        log_file.write(message + '\n')

# Write hyperparameters and model structure
def log_config(model, optimizer, scheduler):
    log_message(f"Hyperparameters and Configuration:")
    log_message(f"Learning Rate: 0.0001")
    log_message(f"Batch Size: {batchsize}")
    log_message(f"Epochs: {epochs}")
    log_message(f"Optimizer: Adam")
    log_message(f"Scheduler: CosineAnnealingLR")
    log_message(f"Model Architecture: {model}")
    log_message(f"Optimizer: {optimizer}")
    log_message(f"Scheduler: {scheduler}")

# Calculate pose error and record log
def log_errors(epoch, step, phase, depth_error, height_error, pitch_error, roll_error, fov_error):
    if phase == "Train":
        log_message(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_dataloader)}] - "
                    f"Depth Error: {depth_error:.4f}, "
                    f"Height Error: {height_error:.4f}, "
                    f"Pitch Error: {pitch_error:.4f}, "
                    f"Roll Error: {roll_error:.4f}, "
                    f"FOV Error: {fov_error:.4f}")
    else:
        log_message(f"Epoch [{epoch+1}/{epochs}] {phase} - "
                    f"Depth Error: {depth_error:.4f}, "
                    f"Height Error: {height_error:.4f}, "
                    f"Pitch Error: {pitch_error:.4f}, "
                    f"Roll Error: {roll_error:.4f}, "
                    f"FOV Error: {fov_error:.4f}"
                    )


def log_loss(epoch, step, total_loss, depth_init_loss, depth_final_loss, pose_init_loss, pose_final_loss, depth_binary_loss, depth_quant_loss_16, depth_quant_loss_64, depth_grad_loss, pose_quant_loss_16, pose_quant_loss_64, loss_p2d):

    log_message(f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_dataloader)}] - "
                f"total_loss: {total_loss:.4f}, "
                f"depth_init_loss: {depth_init_loss.item():.4f}, "
                f"depth_final_loss: {depth_final_loss.item():.4f}, "
                f"depth_grad_loss: {depth_grad_loss.item():.4f}, "
                f"pose_init_loss: {pose_init_loss.item():.4f}, "
                f"pose_final_loss: {pose_final_loss.item():.4f}, "
                f"depth_binary_loss: {depth_binary_loss.item():.4f}, "
                f"depth_quant_loss_16: {depth_quant_loss_16.item():.4f}, "
                f"depth_quant_loss_64: {depth_quant_loss_64.item():.4f}, "
                f"pose_quant_loss_16: {pose_quant_loss_16.item():.4f}, "
                f"pose_quant_loss_64: {pose_quant_loss_64.item():.4f}, "
                f"loss_p2d: {loss_p2d.item():.4f}, "
                )
    


def log_pose_errors(height_errs: torch.Tensor,
                    pitch_errs: torch.Tensor,
                    roll_errs: torch.Tensor,
                    fov_errs: torch.Tensor,
                    depth_errs: torch.Tensor,
                    epoch: int,
                    total_epochs: int,
                    tag: str = "Train") -> None:
    """
    Compute and log MAE, Median, and AUC@1°/5°/10° for aggregated errors over entire dataset or split.

    Args:
        height_errs, pitch_errs, roll_errs, fov_errs, depth_errs: Tensors [N_total]
        epoch: current epoch index (0-based)
        total_epochs: total number of epochs
        tag: arbitrary string label (e.g., "Train" or "Validation")
    """
    # Basic metrics
    mae = lambda errs: errs.mean().item()
    med = lambda errs: errs.median().item()
    # AUC metrics
    aucs_h = compute_auc_per_threshold(height_errs)
    aucs_p = compute_auc_per_threshold(pitch_errs)
    aucs_r = compute_auc_per_threshold(roll_errs)
    aucs_f = compute_auc_per_threshold(fov_errs)

    metrics = {
        'depth': {'MAE': mae(depth_errs), 'Median': med(depth_errs)},
        'height': {'MAE': mae(height_errs), 'Median': med(height_errs), **aucs_h},
        'pitch':  {'MAE': mae(pitch_errs),  'Median': med(pitch_errs),  **aucs_p},
        'roll':   {'MAE': mae(roll_errs),   'Median': med(roll_errs),   **aucs_r},
        'fov':    {'MAE': mae(fov_errs),    'Median': med(fov_errs),    **aucs_f},
    }

    # Build log output
    lines = [f"[{tag}] Epoch [{epoch+1}/{total_epochs}]"]
    lines.append(f"  Depth → MAE: {metrics['depth']['MAE']:.4f}, Median: {metrics['depth']['Median']:.4f}")
    for name in ['height', 'pitch', 'roll', 'fov']:
        m = metrics[name]
        auc_str = ", ".join([f"{k}: {v:.2f}" for k, v in m.items() if k.startswith('AUC')])
        lines.append(
            f"  {name.capitalize():<6} → MAE: {m['MAE']:.4f}, Median: {m['Median']:.4f}, {auc_str}"
        )
    # Append to log file
    log_message("\n".join(lines))