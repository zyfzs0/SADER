import os
import sys
import torch
import numpy as np
import math
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from . import pytorch_ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
first_run = True
class Metric(object):
    def reset(self): pass
    def add(self):   pass
    def value(self): pass


def img_metrics(target, pred, masks=None, var=None, pixelwise=True):
    if target.shape[1]==pred.shape[1]==2*13:
        # print(target[1])
        target = target[:,:13,...]
        pred = pred[:,:13,...]
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    # psnr = 20 * torch.log10(1 / (rmse + 1e-9))
    psnr = 20 * torch.log10(1 / (rmse))
    mae = torch.mean(torch.abs(target - pred))
    
    # spectral angle mapper
    mat = target * pred
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(target * target, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(pred * pred, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1))*torch.tensor(180)/math.pi)

    ssim = pytorch_ssim.ssim(target, pred)

    metric_dict = {'RMSE': rmse.cpu().numpy().item(),
                   'MAE': mae.cpu().numpy().item(),
                   'PSNR': psnr.cpu().numpy().item(),
                   'SAM': sam.cpu().numpy().item(),
                   'SSIM': ssim.cpu().numpy().item()}
    
    if masks is not None:
        # masks: (1, T, H, W)
        # target, pred: (1, C, H, W)
        B, C, H, W = target.shape
        T = masks.shape[1]

        # clone 一份用于指标计算（防止梯度污染或 inplace）
        target_eval = target.clone().detach()
        pred_eval = pred.clone().detach()

        rmse_cloudy_list, rmse_cloudfree_list = [], []
        mae_cloudy_list, mae_cloudfree_list = [], []
        sam_cloudy_list, sam_cloudfree_list = [], []
        ssim_cloudy_list, ssim_cloudfree_list = [], []
        lpips_cloudy_list, lpips_cloudfree_list = [], []
        psnr_cloudy_list, psnr_cloudfree_list = [], []
        masks = masks.float()
        for t in range(T):
            mask_t = masks[:, t, :, :].clamp(0, 1)  # shape (1, H, W)
            mask_t3 = mask_t.repeat(1, C, 1, 1)     # 扩展到通道维
            # print(mask_t3.shape)
            real_B = target_eval.cpu().numpy()
            fake_B = pred_eval.cpu().numpy()
            mask_np = mask_t3.cpu().numpy()

            # ---------- RMSE / MAE ----------
            rmse_cloudy = np.sqrt(np.nanmean(np.square(real_B[mask_np >0.5] - fake_B[mask_np >0.5])))
            rmse_cloudfree = np.sqrt(np.nanmean(np.square(real_B[mask_np <0.5] - fake_B[mask_np <0.5])))
            mae_cloudy = np.nanmean(np.abs(real_B[mask_np>0.5] - fake_B[mask_np>0.5]))
            mae_cloudfree = np.nanmean(np.abs(real_B[mask_np <0.5] - fake_B[mask_np <0.5]))

            rmse_cloudy_list.append(rmse_cloudy)
            rmse_cloudfree_list.append(rmse_cloudfree)
            mae_cloudy_list.append(mae_cloudy)
            mae_cloudfree_list.append(mae_cloudfree)

            # ---------- SAM ----------
            def masked_sam(x, y, m):
                m_expand = m.expand_as(x)
                x_m = x[m_expand]
                y_m = y[m_expand]
                if x_m.numel() == 0:
                    return np.nan
                x_m = x_m.view(-1, C)
                y_m = y_m.view(-1, C)
                num = torch.sum(x_m * y_m, dim=1)
                den = torch.sqrt(torch.sum(x_m * x_m, dim=1)) * torch.sqrt(torch.sum(y_m * y_m, dim=1))
                cos = torch.clamp(num / (den + 1e-8), -1, 1)
                return torch.mean(torch.acos(cos) * 180 / math.pi).item()

            sam_cloudy = masked_sam(target_eval, pred_eval, mask_t.bool())
            sam_cloudfree = masked_sam(target_eval, pred_eval, (~mask_t.bool()))
            sam_cloudy_list.append(sam_cloudy)
            sam_cloudfree_list.append(sam_cloudfree)

            
            def compute_psnr(x, y, mask_np):
                diff = (x - y) ** 2
                mse = np.nanmean(diff[mask_np >0.5]) if np.any(mask_np >0.5) else np.nan
                if mse == 0 or np.isnan(mse):
                    return np.nan
                psnr = 10 * np.log10(1.0 / mse)
                return psnr

            psnr_cloudy = compute_psnr(real_B, fake_B, mask_np)
            psnr_cloudfree = compute_psnr(real_B, fake_B, 1 - mask_np)
            psnr_cloudy_list.append(psnr_cloudy)
            psnr_cloudfree_list.append(psnr_cloudfree)
            # ---------- SSIM ----------
            ssim_cloudy = pytorch_ssim.ssim(target_eval * mask_t3, pred_eval * mask_t3)
            ssim_cloudfree = pytorch_ssim.ssim(target_eval * (1. - mask_t3), pred_eval * (1. - mask_t3))
            ssim_cloudy_list.append(ssim_cloudy.cpu().numpy().item())
            ssim_cloudfree_list.append(ssim_cloudfree.cpu().numpy().item())

            # ---------- LPIPS ----------
            if target_eval.shape[1] in [3, 13]:
                if first_run:
                    loss_fn_alex.to(target_eval.device)
                if target_eval.shape[1] == 13:
                    target_rgb = target_eval[:, [3, 2, 1], ...]
                    pred_rgb = pred_eval[:, [3, 2, 1], ...]
                    mask_rgb = mask_t.repeat(1, 3, 1, 1)
                else:
                    target_rgb, pred_rgb = target_eval, pred_eval
                    mask_rgb = mask_t.repeat(1, 3, 1, 1)

                d_cloudy = loss_fn_alex(target_rgb * mask_rgb * 2 - 1, pred_rgb * mask_rgb * 2 - 1)
                d_cloudfree = loss_fn_alex(target_rgb * (1 - mask_rgb) * 2 - 1, pred_rgb * (1 - mask_rgb) * 2 - 1)

                lpips_cloudy_list.append(d_cloudy.item())
                lpips_cloudfree_list.append(d_cloudfree.item())
            
        # ---------- 聚合结果（跨时间平均） ----------
        metric_dict.update({
            'RMSE_cloudy': np.nanmean(rmse_cloudy_list),
            'RMSE_cloudfree': np.nanmean(rmse_cloudfree_list),
            'MAE_cloudy': np.nanmean(mae_cloudy_list),
            'MAE_cloudfree': np.nanmean(mae_cloudfree_list),
            'SAM_cloudy': np.nanmean(sam_cloudy_list),
            'SAM_cloudfree': np.nanmean(sam_cloudfree_list),
            'SSIM_cloudy': np.nanmean(ssim_cloudy_list),
            'SSIM_cloudfree': np.nanmean(ssim_cloudfree_list),
            'PSNR_cloudy': np.nanmean(psnr_cloudy_list),      
            'PSNR_cloudfree': np.nanmean(psnr_cloudfree_list), 
        })
    # evaluate the (optional) variance maps
    if var is not None:
        error = target - pred
        # average across the spectral dimensions
        se = torch.square(error)
        ae = torch.abs(error)

        # collect sample-wise error, AE, SE and uncertainties
 
        # define a sample as 1 image and provide image-wise statistics
        errvar_samplewise = {'error': error.nanmean().cpu().numpy().item(),
                            'mean ae': ae.nanmean().cpu().numpy().item(),
                            'mean se': se.nanmean().cpu().numpy().item(),
                            'mean var': var.nanmean().cpu().numpy().item()}
        if pixelwise:
            # define a sample as 1 multivariate pixel and provide image-wise statistics
            errvar_samplewise = {**errvar_samplewise, **{'pixelwise error': error.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise ae': ae.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise se': se.nanmean(0).nanmean(0).flatten().cpu().numpy(),
                                                        'pixelwise var': var.nanmean(0).nanmean(0).flatten().cpu().numpy()}}

        metric_dict     = {**metric_dict, **errvar_samplewise}

    if target.shape[1] == pred.shape[1] == 3: # for rgb image
        if first_run:
            loss_fn_alex.to(target.device)
        d = loss_fn_alex(target * 2 - 1, pred * 2 - 1)
        metric_dict['LPIPS'] = d.item()
    elif target.shape[1] == pred.shape[1] == 13: # for single band image
        if first_run:
            loss_fn_alex.to(target.device)
        d = loss_fn_alex(target[:,[3,2,1],...] * 2 - 1, pred[:,[3,2,1],...] * 2 - 1)
        metric_dict['LPIPS'] = d.item()
    return metric_dict

class avg_img_metrics(Metric):
    def __init__(self, cloudfree_cloudy=True):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS']
        self.metrics  += ['error', 'mean se', 'mean ae', 'mean var']
        if cloudfree_cloudy:
            self.metrics += ['RMSE_cloudy', 'MAE_cloudy', 'PSNR_cloud', 'SAM_cloud', 'SSIM_cloud', 'LPIPS_cloud']
                    
        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # skip variables not registered
            if key not in self.metrics: continue
            # filter variables not translated to numpy yet
            if torch.is_tensor(val): continue
            if isinstance(val, tuple): val=val[0]

            # only keep a running mean of non-nan values
            if np.isnan(val): continue
            # if float("inf") == val: continue

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics
    
   
    def extend(self,metrics):
        if isinstance(metrics,avg_img_metrics):
            for key in self.metrics:
                count1 = self.running_nonan_count[key]
                value1 = self.running_img_metrics[key]
                count2 = metrics.running_nonan_count[key]
                value2 = metrics.running_img_metrics[key]
                if count1 != 0 and count2 != 0:
                    self.running_nonan_count[key] = count1 + count2
                    self.running_img_metrics[key] = (count1 * value1 + count2 * value2) / (count1 + count2) 
                elif count1 != 0 and count2 == 0:
                    continue
                elif count1 == 0 and count2 != 0:
                    self.running_nonan_count[key] = count2
                    self.running_img_metrics[key] = value2
                else:
                    self.running_nonan_count[key] = 0
                    self.running_img_metrics[key] = np.nan
        else:
            raise TypeError