import torch
import torch.nn.functional as F
import math



def generate_log_depth(depth_map: torch.Tensor, far: float = 1000.0) -> torch.Tensor:
    """
    Apply logarithmic transformation and normalize to 8-bit grayscale (PyTorch version)
    :param depth_map: Floating point depth map tensor (unit: meters), shape [H, W] or [B, H, W]
    :param far: Far plane distance (default 1000 meters)
    :return: 8-bit logarithmic depth map (uint8 type)
    """
    # device = depth_map.device  # Keep device consistent
    depth_plus_one = depth_map + 1.0  # Prevent log(0) [^original code logic]
    log_depth = torch.log2(depth_plus_one) / torch.log2(torch.tensor(far + 1.0))  # Normalize to [0,1] [7](@ref)
    # log_depth_normalized = (log_depth * 255).to(torch.uint8)  # Convert to 8-bit grayscale [^original code logic]
    return log_depth


def calculate_theta_with_roll_and_depth(
    H_max: int, W_max: int,
    fov_deg: torch.Tensor,  # [B]
    pitch_deg: torch.Tensor,  # [B]
    roll_deg: torch.Tensor,  # [B]
    h: torch.Tensor         # [B]
) -> torch.Tensor:
    """
    Compute depth image per batch with FOV, pitch, roll, height vectors.
    Returns tensor [B,1,H_max,W_max]
    """
    B = fov_deg.shape[0]

    # to radians and reshape for broadcasting
    fov_rad = torch.deg2rad(fov_deg).view(B, 1, 1)
    pitch_rad = torch.deg2rad(pitch_deg).view(B, 1, 1)
    roll_rad = torch.deg2rad(roll_deg).view(B, 1, 1)

    # pixel grid H, W without unsupported indexing arg
    center_h = H_max / 2.0 - 0.5
    center_w = W_max / 2.0 - 0.5
    hr = torch.arange(H_max, device=fov_rad.device)
    wr = torch.arange(W_max, device=fov_rad.device)
    i, j = torch.meshgrid(hr, wr)  # legacy defaults to 'ij'

    H = (i - center_h).unsqueeze(0).expand(B, -1, -1)
    W = (j - center_w).unsqueeze(0).expand(B, -1, -1)

    # compute per-pixel view angles
    tan_fov_2 = torch.tan(fov_rad / 2)
    theta_x_rad = torch.atan(2 * H * tan_fov_2 / H_max)
    theta_y_rad = torch.atan(2 * W * tan_fov_2 / W_max)

    # rotate pixels by roll
    cos_r = torch.cos(roll_rad)
    sin_r = torch.sin(roll_rad)
    new_H = cos_r * H + sin_r * W
    new_W = -sin_r * H + cos_r * W

    # get direction slope magnitude
    tan_theta = torch.sqrt(torch.tan(theta_x_rad)**2 + torch.tan(theta_y_rad)**2)
    denom = torch.sqrt(new_H**2 + new_W**2)
    tan_theta_x = new_H * tan_theta / denom
    tan_theta_y = new_W * tan_theta / denom
    theta_x_rad_new = torch.atan(tan_theta_x)
    theta_y_rad_new = torch.atan(tan_theta_y)

    # combine with pitch for vertical angle
    theta_sum = pitch_rad + theta_y_rad_new

    # compute depth
    h_b = h.view(B, 1, 1)
    depth = h_b * torch.sqrt(torch.tan(theta_x_rad_new)**2 + torch.tan(theta_sum)**2 + 1)

    # handle invalid (looking horizontally or upward)
    invalid_mask = theta_sum >= (math.pi / 2)
    depth = torch.where(invalid_mask, torch.full_like(depth, float('inf')), depth)
    depth = torch.clamp(depth, max=1000.0)

    # log depth normalization
    depth_log = generate_log_depth(depth)

    # expand channel dim and rotate
    depth_image = depth_log.unsqueeze(1)  # [B,1,H,W]
    depth_image = torch.rot90(depth_image, k=1, dims=[2, 3])

    return depth_image


def pose_to_log_depth(pose):

    height = pose[:, 0] * 90 + 10     # 10, 100
    pitch  = pose[:, 1] * 110 - 100   # -100, 10  
    roll   = pose[:, 2] * 60 - 30     # -30, 30
    fov    = pose[:, 3] * 90 + 30     # 30, 120

    log_depth = calculate_theta_with_roll_and_depth(512, 512, fov, pitch + 90, roll, height)

    return log_depth



# Calculate the error for each parameter (absolute error)
def calculate_pose_errors(predicted_pose, true_pose, predicted_depth, true_depth):

    # Error between predicted value and true value (absolute error)
    height_error = torch.abs(predicted_pose[0] - true_pose[0])
    pitch_error = torch.abs(predicted_pose[1] - true_pose[1])
    roll_error = torch.abs(predicted_pose[2] - true_pose[2])
    fov_error = torch.abs(predicted_pose[3] - true_pose[3])

    depth_error = torch.abs(predicted_depth - true_depth) * 255

    height_error, pitch_error, roll_error, fov_error = height_error * 90, (pitch_error * 110 ), (roll_error * 60), (fov_error * 90)


    return height_error, pitch_error, roll_error, fov_error, depth_error




def calculate_pose_errors(predicted_pose: torch.Tensor,
                          true_pose: torch.Tensor,
                          predicted_depth: torch.Tensor,
                          true_depth: torch.Tensor) -> tuple:
    """
    Compute absolute errors for pose parameters and depth per sample.
    Scaling:
        - height in degrees: *90
        - pitch in degrees: *110
        - roll in degrees: *60
        - fov in degrees: *90
        - depth: rescaled by 255 factor
    Returns:
        height_errs, pitch_errs, roll_errs, fov_errs, depth_errs (each Tensor [B])
    """
    height_errs = torch.abs(predicted_pose[:, 0] - true_pose[:, 0]) * 90
    pitch_errs  = torch.abs(predicted_pose[:, 1] - true_pose[:, 1]) * 110
    roll_errs   = torch.abs(predicted_pose[:, 2] - true_pose[:, 2]) * 60
    fov_errs    = torch.abs(predicted_pose[:, 3] - true_pose[:, 3]) * 90
    depth_errs  = torch.abs(predicted_depth - true_depth) * 255
    return height_errs, pitch_errs, roll_errs, fov_errs, depth_errs


def compute_auc_per_threshold(errs: torch.Tensor, max_thresholds: list = [1.0, 5.0, 10.0]) -> dict:
    """
    Compute normalized AUC of recall-threshold curve up to each max threshold.

    For each T in max_thresholds:
      - thresholds = [t for t in max_thresholds if t <= T]
      - include 0 and T in xs
      - compute recalls at those points
      - integrate piecewise and normalize by T
    Returns:
        dict mapping 'AUC@{T}°' to value
    """
    aucs = {}
    # full thresholds and recalls
    all_ths = max_thresholds
    recalls = [(errs <= t).float().mean().item() for t in all_ths]
    # prepend 0
    xs_full = [0.0] + all_ths
    ys_full = [0.0] + recalls
    # compute prefix areas
    prefix_areas = [0.0]
    for i in range(len(xs_full) - 1):
        dx = xs_full[i+1] - xs_full[i]
        area = 0.5 * (ys_full[i] + ys_full[i+1]) * dx
        prefix_areas.append(prefix_areas[-1] + area)
    # for each threshold compute normalized area
    for idx, T in enumerate(all_ths, start=1):
        area_T = prefix_areas[idx]
        aucs[f"AUC@{int(T) if T.is_integer() else T}°"] = area_T / T
    return aucs



def compute_depth_metrics(pred, gt):

    pred *= 255
    gt   *= 255

    # Avoid 0 or negative values
    pred = torch.clamp(pred, min=1e-3)
    gt   = torch.clamp(gt,   min=1e-3)

    # Proportional error
    ratio = torch.max(pred/gt, gt/pred)

    δ1 = (ratio < 1.25   ).float().mean()
    δ2 = (ratio < 1.25**2).float().mean()
    δ3 = (ratio < 1.25**3).float().mean()

    # Absolute relative error
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    # RMSE
    rmse = torch.sqrt(torch.mean((gt - pred)**2))

    # log10
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(gt)))


    return {
        'δ1': δ1.item(), 'δ2': δ2.item(), 'δ3': δ3.item(),
        'AbsRel': abs_rel.item(), 'RMSE': rmse.item(),
        'log10': log10.item(),
    }