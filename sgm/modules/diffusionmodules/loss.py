from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config, tools_scale, tools_scale2,sen_mtc_scale_01
from .denoiser import Denoiser
from .sigma2st import Sigma2St
from torchvision.transforms import v2
import cv2

class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

class ResidualDiffusionLoss(StandardDiffusionLoss):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, sigma2st, input, mu, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w)
    
    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + mu + noise * sigmas_bc
        return noised_input



class ResidualDiffusion_MaskLoss_NoScale(ResidualDiffusionLoss):
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    
    def get_loss(self, model_output, target, w, mu, eps:float = 1e-8):
        
        target_expanded = target  # (b, c, h, w)
        diff = torch.abs(mu - target_expanded)
        # 计算每个通道的空间最小/最大值 (保持 b, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        # sequence_len_tensor = torch.tensor(1, dtype=torch.float32)  # 转换为浮点张量
        # max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        attention_mask_scaled = attention_mask
        threshold = self.threshold  # 可调整的阈值
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w,cloud_image)



class ResidualDiffusion_MaskLoss(ResidualDiffusionLoss):
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    
    def get_loss(self, model_output, target, w, mu, eps:float = 1e-8):
        
        target_expanded = target  # (b, c, h, w)
        diff = torch.abs(mu - target_expanded)
        # 计算每个通道的空间最小/最大值 (保持 b, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(1, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        attention_mask_scaled = torch.sigmoid(attention_mask) * 1.0 /max_scale
        threshold = self.threshold  # 可调整的阈值
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w,cloud_image)
    

class ResidualDiffusion_AlphaMaskLoss(ResidualDiffusionLoss):
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    
    def get_loss(self, model_output, target, w, mu, eps:float = 1e-8):
        sequence_len = 1
        target_expanded = target  # (b, c, h, w)
        diff0= torch.abs(mu - target_expanded)
        diff = (diff0)/(1.0-target_expanded+eps)
        # 计算每个通道的空间最小/最大值 (保持 b, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(1, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        # attention_mask_scaled = torch.sigmoid(attention_mask) * 1.0 /max_scale
        threshold = self.threshold  # 可调整的阈值
        positive_mask = (attention_mask > (1.0-threshold))
        attention_mask_scaled = torch.where(
            positive_mask,
            torch.sigmoid(attention_mask * sequence_len) * 1.0 / max_scale,
            torch.zeros_like(attention_mask)  # 其余位置置0
        )
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w,cloud_image)


class ResidualDiffusion_Alpha_YUV(ResidualDiffusionLoss):
    def __init__(self, rgb2yuv_config=None,yuv_multiple=1.0,attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
        self.rgb2yuv = instantiate_from_config(rgb2yuv_config)
        self.yuv_multiple = yuv_multiple
    
    def get_loss(self, model_output, target, w, mu, eps:float = 1e-8):
        sequence_len = 1
        target_expanded = target  # (b, c, h, w)
        yuv_output = self.rgb2yuv(model_output)
        vuy_target = self.rgb2yuv(target_expanded)
        diff0= torch.abs(mu - target_expanded)
        diff = (diff0)/(1.0-target_expanded+eps)
        # 计算每个通道的空间最小/最大值 (保持 b, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(1, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        # attention_mask_scaled = torch.sigmoid(attention_mask) * 1.0 /max_scale
        threshold = self.threshold  # 可调整的阈值
        positive_mask = (attention_mask > (1.0-threshold))
        attention_mask_scaled = torch.where(
            positive_mask,
            torch.sigmoid(attention_mask * sequence_len) * 1.0 / max_scale,
            torch.zeros_like(attention_mask)  # 其余位置置0
        )
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 )+self.yuv_multiple*(yuv_output-yuv_target)**2)).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs()+self.yuv_multiple*(yuv_output-yuv_target).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        st = sigma2st(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2])
                if self.n_frames is not None
                else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)

        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w,cloud_image)
    

class TemporalResidualDiffusionLoss(ResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w)



class TemporalResidualDiffusion_MaskLoss_NoScale(TemporalResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    def get_loss(self, model_output, target, w, mu, eps: float = 1e-8):
        sequence_len = mu.shape[1]
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (b, seq, c, h, w)
        diff = torch.abs(mu - target_expanded)
        # 计算每个通道的空间最小/最大值 (保持 b, seq, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, seq, c, h, w)
        # 沿 seq 维度取合
        attention_mask = attention_mask.mean(dim=1)   # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        #sequence_len_tensor = torch.tensor(sequence_len, dtype=torch.float32)  # 转换为浮点张量
        # max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        #attention_mask_scaled = torch.sigmoid(attention_mask*sequence_len) * 1.0 /max_scale
        attention_mask_scaled = attention_mask
        threshold = self.threshold  # 可调整的阈值
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
    
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w, cloud_image)  


class TemporalResidualDiffusion_MaskLoss(TemporalResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    def get_loss(self, model_output, target, w, mu, eps: float = 1e-8):
        sequence_len = mu.shape[1]
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (b, seq, c, h, w)
        diff = torch.abs(mu - target_expanded)
        # 计算每个通道的空间最小/最大值 (保持 b, seq, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, seq, c, h, w)
        # 沿 seq 维度取合
        attention_mask = attention_mask.mean(dim=1)   # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(1, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        attention_mask_scaled = torch.sigmoid(attention_mask) * 1.0 /max_scale
        threshold = self.threshold  # 可调整的阈值
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
    
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w, cloud_image)   
    

class TemporalResidualDiffusion_MaskLoss_Seq(TemporalResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    def get_loss(self, model_output, target, w, mu, eps: float = 1e-8):
        sequence_len = mu.shape[1]
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (b, seq, c, h, w)
        diff = torch.abs(mu - target_expanded)
        # 计算每个通道的空间最小/最大值 (保持 b, seq, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, seq, c, h, w)
        # 沿 seq 维度取合
        attention_mask = attention_mask.mean(dim=1)   # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(sequence_len, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        attention_mask_scaled = torch.sigmoid(attention_mask*sequence_len) * 1.0 /max_scale
        threshold = self.threshold  # 可调整的阈值
        unattention_mask_scaled = torch.where(unattention_mask >= threshold, 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
    
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w, cloud_image)  
    
def stretch_rgb_sen2(img, low=2, high=98):
    """
    img: torch.Tensor (3,H,W) in any range, any device
    returns: torch.Tensor (3,H,W) scaled into [0,1]
    """
    img = img.float()
    out = torch.zeros_like(img)
    for c in range(3):
        ch = img[c]
        flat = ch.flatten()
        k_low = int((low / 100.0) * (flat.numel() - 1)) + 1
        k_high = int((high / 100.0) * (flat.numel() - 1)) + 1
        vmin = torch.kthvalue(flat, k_low).values
        vmax = torch.kthvalue(flat, k_high).values
        ch = (ch - vmin) / (vmax - vmin + 1e-6)
        ch = torch.clamp(ch, 0.0, 1.0)
        out[c] = ch
    return out
def save_tensor_as_rgb(path, tensor):
    """
    tensor: (C,H,W), value in [0,1]
    Saves as RGB image using cv2 (converts to BGR automatically)
    """
    import cv2
    x = tensor.detach().cpu().float().numpy()   # -> numpy
    x = x.transpose(1, 2, 0)                    # (H,W,C)  RGB
    x = (x * 255).clip(0,255).astype(np.uint8)  # 归一化
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)      # ⭐重点：RGB → BGR
    cv2.imwrite(path, x)
'''
save_tensor_as_rgb("output.png", scale(target)[0])
save_tensor_as_rgb("alpha0.png", diff[0][0])
save_tensor_as_rgb("alpha1.png", diff[0][1])
save_tensor_as_rgb("alpha2.png", diff[0][2])
save_tensor_as_rgb("attention.png", attention_mask[0])
save_tensor_as_rgb("unattention.png", unattention_mask[0])
save_tensor_as_rgb("attention_scaled.png", attention_mask_scaled[0])
save_tensor_as_rgb("unattention_scaled.png", unattention_mask_scaled[0])
Y = yuv_img[0]        # (H,W)
U = yuv_img[1]
V = yuv_img[2]
# Y, U, V 分通道灰度图
cv2.imwrite("y_channel.png", (Y.cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("u_channel.png", (U.cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("v_channel.png", (V.cpu().numpy()*255).astype(np.uint8))
'''
    
class TemporalResidualDiffusion_AlphaMaskLoss_Seq(TemporalResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)
    def __init__(self, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
    def get_loss(self, model_output, target, w, mu, eps: float = 1e-8):
        
        '''
        # TODO:loss函数需要加亮度损失和多个实验输出云层
        '''
        sequence_len = mu.shape[1]
        # scale = sen_mtc_scale_01()
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (b, seq, c, h, w)
        diff0 = torch.abs(mu - target_expanded)
        diff0_m = diff0.mean(dim=1)
        diff0_m_b= torch.where(diff0_m > 0.1, 1.0, 0.0)
        diff0_m_w = torch.where(diff0_m <=0.1, 1.0, 0.0)
        diff = (diff0)/(1.0- target_expanded+eps)
        # 计算每个通道的空间最小/最大值 (保持 b, seq, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, seq, c, h, w)
        # 沿 seq 维度取合
        attention_mask = attention_mask.mean(dim=1)   # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(sequence_len, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        # attention_mask_scaled = torch.sigmoid(attention_mask*sequence_len) * 1.0 /max_scal3
        threshold = self.threshold  # 可调整的阈值
        positive_mask = (attention_mask > (1.0-threshold))
        # 仅对大于0的位置应用变换，其余保持0
        attention_mask_scaled = torch.where(
            positive_mask,
            torch.sigmoid(attention_mask * sequence_len) * 1.0 / max_scale,
            torch.zeros_like(attention_mask)  # 其余位置置0
        )
        unattention_mask_scaled = torch.where(unattention_mask >threshold , 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            return torch.mean(
                (w * (self.attention_mutiple*(attention_mask_scaled.abs() *(model_output - target)**2 ) +self.unattention_multiple*(unattention_mask_scaled.abs() *(model_output - target)**2 ))).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (self.attention_mutiple*((model_output - target)*attention_mask_scaled).abs()+ self.unattention_multiple*((model_output - target)*unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
    
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w, cloud_image)  

'''
save_tensor_as_rgb("./loss_pic/output.png",(tools_scale2(target[0])))
save_tensor_as_rgb("./loss_pic/mu0.png", (tools_scale2(mu[0][0])))
save_tensor_as_rgb("./loss_pic/mu1.png", (tools_scale2(mu[0][1])))
save_tensor_as_rgb("./loss_pic/mu2.png", (tools_scale2(mu[0][2])))
save_tensor_as_rgb("./loss_pic/alpha0.png", diff[0][0])
save_tensor_as_rgb("./loss_pic/alpha1.png", diff[0][1])
save_tensor_as_rgb("./loss_pic/alpha2.png", diff[0][2])
save_tensor_as_rgb("./loss_pic/attention.png", attention_mask[0])
save_tensor_as_rgb("./loss_pic/unattention.png", unattention_mask[0])
save_tensor_as_rgb("./loss_pic/attention_scaled.png", attention_mask_scaled[0])
save_tensor_as_rgb("./loss_pic/unattention_scaled.png", unattention_mask_scaled[0])
Y = yuv_img[0]        # (H,W)
U = yuv_img[1]
V = yuv_img[2]
# Y, U, V 分通道灰度图
cv2.imwrite("./loss_pic/y_channel.png", (yuv_output[0][0].detach().cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("./loss_pic/u_channel.png", (yuv_output[0][1].detach().cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("./loss_pic/v_channel.png", (yuv_output[0][2].detach().cpu().numpy()*255).astype(np.uint8))
# Y, U, V 分通道灰度图
cv2.imwrite("./loss_pic/y_target.png", (yuv_target[0][0].detach().cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("./loss_pic/u_target.png", (yuv_target[0][1].detach().cpu().numpy()*255).astype(np.uint8))
cv2.imwrite("./loss_pic/v_target.png", (yuv_target[0][2].detach().cpu().numpy()*255).astype(np.uint8))
'''
class TemporalResidualDiffusion_Alpha_YUV(TemporalResidualDiffusionLoss):    
    # def __init__(self, get_skip_index_config, *args, **kwargs):
    #     self.get_skip_index = instantiate_from_config(get_skip_index_config)
    #     super().__init__(*args, **kwargs)
    def __init__(self,rgb2yuv_config=None,yuv_multiple=1.0, attention_multiple = 1.0,unattention_multiple = 1.0, threshold = 0.99,train_mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mutiple = attention_multiple
        self.unattention_multiple = unattention_multiple
        self.threshold = threshold
        self.rgb2yuv = instantiate_from_config(rgb2yuv_config)
        self.yuv_multiple = yuv_multiple
        self.train_mask = train_mask
    
    def get_loss(self, model_output, target, w, mu, eps: float = 1e-8):
        
        '''
        # TODO:loss函数需要加亮度损失和多个实验输出云层
        '''
        sequence_len = mu.shape[1]
        # scale = sen_mtc_scale_01()
        yuv_output = self.rgb2yuv(model_output)
        target_expanded = target.unsqueeze(1).expand_as(mu)  # (b, seq, c, h, w)
        yuv_target = self.rgb2yuv(target)
        diff0 = torch.abs(mu - target_expanded)
        diff0_m = diff0.mean(dim=1)
        diff0_m_b= torch.where(diff0_m > 0.1, 1.0, 0.0)
        diff0_m_w = torch.where(diff0_m <=0.1, 1.0, 0.0)
        diff = (diff0)/(1.0- target_expanded+eps)
        # 计算每个通道的空间最小/最大值 (保持 b, seq, c 维度)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 最小-最大归一化 (按通道独立处理)
        attention_mask = (diff - min_val) / (max_val - min_val + eps)  # (b, seq, c, h, w)
        # 沿 seq 维度取合
        attention_mask = attention_mask.mean(dim=1)   # (b, c, h, w)
        unattention_mask = 1.0 - attention_mask               # (b, c, h, w)
        sequence_len_tensor = torch.tensor(1.0, dtype=torch.float32)  # 转换为浮点张量
        max_scale = 1.0 * torch.sigmoid(sequence_len_tensor)
        # attention_mask_scaled = torch.sigmoid(attention_mask*sequence_len) * 1.0 /max_scal3
        threshold = self.threshold  # 可调整的阈值
        positive_mask = (attention_mask > (1.0-threshold))
        # 仅对大于0的位置应用变换，其余保持0
        attention_mask_scaled = torch.where(
            positive_mask,
            torch.sigmoid(attention_mask * sequence_len) * 1.0 / max_scale,
            torch.zeros_like(attention_mask)  # 其余位置置0
        )
        unattention_mask_scaled = torch.where(unattention_mask >threshold , 1.0, 0.0)
        assert attention_mask_scaled.size() == target.size(), f"Shape mismatch: attention_mask_scaled {attention_mask_scaled.size()} != target {target.size()}"
        assert unattention_mask_scaled.size() == target.size(), f"Shape mismatch: unattention_mask_scaled {unattention_mask_scaled.size()} != target {target.size()}"
        if self.loss_type == "l2":
            # 计算主要损失（attention + unattention）
            main_loss = torch.mean(
                (w * (self.attention_mutiple * (attention_mask_scaled.abs() * (model_output - target)**2) + 
                    self.unattention_multiple * (unattention_mask_scaled.abs() * (model_output - target)**2))).reshape(target.shape[0], -1), 1
            )
            # 计算YUV损失
            yuv_loss = torch.mean(
                (w * self.yuv_multiple * (yuv_output - yuv_target)**2).reshape(target.shape[0], -1), 1
            )
            # 返回平均损失
            return main_loss + yuv_loss
        elif self.loss_type == "l1":
            # 计算主要损失（attention + unattention）
            main_loss = torch.mean(
                (w * (self.attention_mutiple * ((model_output - target) * attention_mask_scaled).abs() + 
                    self.unattention_multiple * ((model_output - target) * unattention_mask_scaled).abs())).reshape(target.shape[0], -1), 1
            )
            # 计算YUV损失
            yuv_loss = torch.mean(
                (w * self.yuv_multiple * (yuv_output - yuv_target).abs()).reshape(target.shape[0], -1), 1
            )
            # 返回平均损失
            return main_loss + yuv_loss
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
    
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        sigma2st: Sigma2St,
        input: torch.Tensor,
        mu: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        sigmas = self.sigma_sampler(input.shape[0]).to(input).view(input.shape[0])
        st = sigma2st(sigmas)
        input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # if self.train_mask:
        #     mask = batch[self.train_mask].float()
        #     mask_expanded = mask.unsqueeze(2)  # (b,t,1,h,w)
        #     noised_input = noised_input * mask_expanded + cloud_image * (1.0 - mask_expanded)
        
        noise = torch.randn_like(input)
        # noise = torch.randn_like(input)
        # noise = noise.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        # input = input.unsqueeze(dim=1).repeat(1,mu.shape[1],1,1,1)
        sigmas_bc = append_dims(sigmas, input.ndim)
        st_bc = append_dims(st, input.ndim)
        cloud_image = mu.clone()
        mu *= ((1.0 - st_bc) / st_bc)
        noised_input = self.get_noised_input(sigmas_bc, noise, input, mu)
        # if self.train_mask:
        #     mask = batch[self.train_mask].float()
        #     mask_expanded = mask.unsqueeze(2)  # (b,t,1,h,w)
        #     noised_input = noised_input * mask_expanded + cloud_image * (1.0 - mask_expanded)
            # print("mask_expanded:",mask_expanded.shape)
        # skip_index = self.get_skip_index(batch)
        model_output = denoiser(
            network, noised_input, sigmas, cond, st, **additional_model_inputs
        )
        input = input[:,0,...] # repeat, so that we only need to use the index 0
        w = append_dims(self.loss_weighting(sigmas, st, int(mu.shape[1])), input.ndim)
        # w = append_dims(self.loss_weighting(sigmas, st), input.ndim)
        return self.get_loss(model_output, input, w, cloud_image)  
    

