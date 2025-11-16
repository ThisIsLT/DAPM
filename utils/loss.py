import torch


def scale_invariant_loss(pred, gt, λ=0.85, a=10.0, eps=1e-3):

    """
    pred: [B, H, W] predicted depth
    gt:   [B, H, W] ground truth depth
    λ:    lambda parameter, 0.85
    α:    scaling factor, 10
    """
    # Avoid log(0)
    pred = torch.clamp(pred, min=eps)
    gt   = torch.clamp(gt,   min=eps)

    # Calculate log difference for each pixel
    g = torch.log(pred) - torch.log(gt)   # [B, H, W]
    B, H, W = g.shape

    # Merge pixel dimensions
    g = g.view(B, -1)  # [B, T], T = H*W

    # Calculate for each sample separately
    # mean of g^2
    term1 = torch.mean(g ** 2, dim=1)             # [B]
    # square of mean g
    term2 = (torch.mean(g, dim=1) ** 2)           # [B]

    # SI loss per sample
    loss_per_sample = torch.sqrt(term1 - λ * term2 + eps)  # [B]

    # Batch average and multiply by α
    loss = a * torch.mean(loss_per_sample)
    return loss






def laplacian_processing(input_tensor):

    device = input_tensor.device    
    batch, in_channels, h, w = input_tensor.shape

    base_kernel = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=torch.float32)
    

    kernel = base_kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)  # [C,1,3,3]

    conv = nn.Conv2d(in_channels=in_channels,
                     out_channels=in_channels,  
                     kernel_size=3,
                     padding=1,               
                     groups=in_channels,       
                     bias=False).to(device)
    
    with torch.no_grad():
        conv.weight = nn.Parameter(kernel.to(device))
        conv.weight.requires_grad_(False)  
    
    laplacian = conv(input_tensor)

    return laplacian + 1


def gard_weight_loss(pred, gt):

    gt_grad = laplacian_processing(gt)
    pred_grad = laplacian_processing(pred)

    gard_mse_loss = (gt_grad - pred_grad) ** 2
    gard_weight_mse_loss = ((gt - pred) ** 2) * gt_grad
    gard_mse_loss = gard_mse_loss.mean()
    gard_weight_mse_loss = gard_weight_mse_loss.mean()

    # print(gard_mse_loss, gard_weight_mse_loss)
    gard_weight_loss = gard_mse_loss + gard_weight_mse_loss

    return gard_weight_loss