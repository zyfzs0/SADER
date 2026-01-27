import numpy as np
import torch

def MAE(pred, true, mask=None):
    """Mean Absolute Error, only on masked positions (mask==0)"""
    if mask is not None:
        mask = mask.repeat(1, 1, 1, true.shape[3])  # broadcast to channels
        pred, true = pred[mask == 0], true[mask == 0]
    return np.mean(np.abs(pred - true))

def MSE(pred, true, mask=None):
    """Mean Squared Error, only on masked positions (mask==0)"""
    if mask is not None:
        mask = mask.repeat(1, 1, 1, true.shape[3])
        pred, true = pred[mask == 0], true[mask == 0]
    return np.mean((pred - true) ** 2)

def RMSE(pred, true, mask=None):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true, mask))

def PSNR(pred, true, max_val=255.0, mask=None):
    """Peak Signal-to-Noise Ratio on masked positions"""
    mse = MSE(pred, true, mask)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def SSIM(pred, true, mask=None, max_val=255.0, eps=1e-8):
    """SSIM only on masked positions"""
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true)
    if mask is not None and isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    pred = pred.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    true = true.permute(0, 3, 1, 2).float()

    if mask is not None:
        mask = mask.permute(0, 3, 1, 2).float()
        mask = mask.repeat(1, pred.shape[1], 1, 1)  # broadcast to channels
        mask_sum = mask.sum(dim=[2, 3], keepdim=True) + eps
        mu_pred = (pred * (1 - mask)).sum(dim=[2, 3], keepdim=True) / mask_sum
        mu_true = (true * (1 - mask)).sum(dim=[2, 3], keepdim=True) / mask_sum
    else:
        mu_pred = pred.mean(dim=[2, 3], keepdim=True)
        mu_true = true.mean(dim=[2, 3], keepdim=True)

    sigma_pred = ((pred - mu_pred) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_true = ((true - mu_true) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_xy = ((pred - mu_pred) * (true - mu_true)).mean(dim=[2, 3], keepdim=True)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim = ((2 * mu_pred * mu_true + C1) * (2 * sigma_xy + C2)) / \
           ((mu_pred ** 2 + mu_true ** 2 + C1) * (sigma_pred + sigma_true + C2))

    return ssim.mean().item()

def R2(pred, true, mask=None):
    """Coefficient of determination (R² score) on masked positions"""
    if mask is not None:
        mask = mask.repeat(1, 1, 1, true.shape[3])
        pred, true = pred[mask == 0], true[mask == 0]
    true_mean = np.mean(true)
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - true_mean) ** 2)
    return 1 - numerator / denominator if denominator != 0 else 0.0

def metric(pred, true, mask=None):
    """
    Return metrics only on masked points
    pred, true: (B, H, W, C)
    mask: (B, H, W, 1)
    """
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    psnr = PSNR(pred, true, 255.0, mask)
    ssim = SSIM(pred, true, mask)
    r2 = R2(pred, true, mask)
    return mae, mse, rmse, psnr, ssim, r2
