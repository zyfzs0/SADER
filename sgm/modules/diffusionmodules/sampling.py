from typing import Dict, Union
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm
import cv2
from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config, tools_scale, tools_scale2, sen_mtc_scale_01, scale_01_from_minus1_1
from torchvision.transforms import v2
DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}
import pickle
import time
import itertools
# scale = sen_mtc_scale_01()

class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class BaseResidualDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, st, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st)
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepResidualDiffusionSampler(BaseResidualDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d

class ResidualEDMSampler(SingleStepResidualDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.sigma2st = None
        
    def set_sigma2st(self, sigma2st):
        self.sigma2st = sigma2st
        
    def prepare_sampling_loop(self, x, mu, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)
        st0 = self.sigma2st(sigmas[0])
        # x = mu + sigmas[0] * st * x
        x = ((1 - st0) / st0) * mu + sigmas[0] * x
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0):
        st = self.sigma2st(sigma) 
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat) / st_hat - (1 - st) / st) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc)
        # d = - (x - mu) - 2 * st_hat_bc * denoised + 2 * x 
        # d = - st_hat_bc * x + mu - (denoised - x) / sigma_hat_bc
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        # d = - st_hat_bc * (x - mu)  - denoised * st_hat_bc / sigma_hat_bc + x / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            x, denoised = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
            )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.clone().detach()))
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x, others

class ResidualEulerEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step
    


class ResidualEulerEDMSampler_Repaint(ResidualEulerEDMSampler):
    def __init__(
        self, repaint_steps = 0, repaint_schurn = 0.0,threshold_diff=0.01, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.repaint_steps = repaint_steps
        self.repaint_schurn = repaint_schurn
        self.threshold_diff = threshold_diff

    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step_inner(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
    
    def repaint_step_inner(self, sigma, next_sigma, denoiser, x,x0, mu, cond, uc=None, gamma=0.0,eps_d = 1e-8):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, False)
        '''
        先统计 x0和denoised差值,选择平均数以上的记为x_temp
        在计算x_temp区域内部,计算x0和mu差值,denoised和mu差值,取差值大的地方mask作为1
        其余mask全部为0
        mask为1的地方用denoised, 0的地方用x0
        '''
        assert denoised.size() == x0.size(), f"Shape mismatch: denoised {denoised.size()} != x0 {x0.size()}"
        # 计算x0和denoised的差值（绝对值）
        diff = torch.abs(x0 - denoised)  # (b, c, h, w)
        
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        # 对每个通道计算差值的平均值（在后两个维度h,w上取平均）
        channel_means = torch.mean(diff, dim=(-2, -1), keepdim=True)  # (b, c, 1, 1)
        normalized_diff = (diff - min_val) / (max_val - min_val + eps_d)
        # 创建临时mask（x_temp），标记差值大于平均值的区域
        # x_temp = torch.logical_and(
        #     diff >= channel_means,          # 条件1：比均值大
        #     normalized_diff > self.threshold_diff # 条件2：归一化差值 > 阈值
        # ).float()  # 转为 0/1 掩码
        x_temp = (normalized_diff > self.threshold_diff).float()  # 仅保留第二个条件
        temp_mask = (1.0 - x_temp)
        mu_masked = mu * temp_mask
        valid_pixels = temp_mask.sum(dim=(-2, -1), keepdim=True)
        valid_pixels = torch.clamp(valid_pixels, min=1)  # 防止除以 0
        
        # mu_m =torch.mean(mu, dim=(-2,-1), keepdim=True) 
        mu_m = mu_masked.sum(dim=(-2, -1), keepdim=True) / valid_pixels
        # 计算x0和mu的差值
        diff_x0_mu = torch.abs(x0 - mu_m)  # (b, c, h, w)
        # 计算denoised和mu的差值
        diff_denoised_mu = torch.abs(denoised - mu_m)  # (b, c, h, w)
        
        # 在x_temp区域内比较两个差值
        # 首先创建x_temp区域的掩码
        x_temp_mask = (x_temp > 0)  # bool mask
        
        # 初始化最终mask
        mask = torch.zeros_like(x0)  # (b,  c, h, w)
        
        # 在x_temp区域内，比较两个差值，取较大的那个
        # 对于x_temp为1的区域，如果diff_denoised_mu < diff_x0_mu，则mask设为1
        mask[x_temp_mask] = (diff_denoised_mu[x_temp_mask] < diff_x0_mu[x_temp_mask]).float()
        mask *= 1.0
        # 生成最终结果：mask为1的地方用denoised，0的地方用x0
        result = mask * denoised + (1.0 - mask) * x0
        
        
        return x, result
        
        
    def get_repaint_allsteps(self):
        return self.repaint_steps 
    
    def repaint_step(self, glb_step,num_sigmas,sigma, next_sigma, denoiser, x, mu, cond, uc=None , gamma =0.0, 
                     return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        re_gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.repaint_schurn / (num_sigmas - 1)
                if self.s_tmin <= sigma <= self.s_tmax
                else 0.0
            )
        if return_attn:
            x, denoised = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        else:
            x, denoised = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        if return_attn:
            denoised, attn = denoised
        base_denoised = denoised
        x_next = x
        # st0 = self.sigma2st(sigmas[0])
        # # x = mu + sigmas[0] * st * x
        # x = ((1 - st0) / st0) * mu + sigmas[0] * x  
        repaint_allsteps = self.get_repaint_allsteps()
        if glb_step != 0:
            for i in range(repaint_allsteps):
                if i == 0:
                    st0 = self.sigma2st(sigma)
                    assert base_denoised.size() == mu.size(), f"Shape mismatch: base_denoised {base_denoised.size()} != mu {mu.size()}"
                    noise = torch.randn_like(base_denoised)
                    x_r = ((1 - st0) / st0) * mu +base_denoised + sigma * noise 
                else :
                    st0 = self.sigma2st(sigma)
                    assert denoised.size() == mu.size(), f"Shape mismatch: denoised {denoised.size()} != mu {mu.size()}"
                    _denoised_other = denoised
                    noise = torch.randn_like(_denoised_other)
                    x_r = ((1 - st0) / st0) * mu +_denoised_other + sigma * noise
                x_0 = denoised
                x_prand,new_denoised = self.repaint_step_inner(
                    sigma,
                    next_sigma,
                    denoiser,
                    x_r,
                    x_0,
                    mu,
                    cond,
                    uc,
                    re_gamma,
                )
                
                denoised = new_denoised
                x_next = x_prand
            
        _denoised = denoised
        assert _denoised.size() == mu.size(), f"Shape mismatch: _denoised {_denoised.size()} != mu {mu.size()}"
        assert x_next.size() == mu.size(), f"Shape mismatch: x_next {x_next.size()} != mu {mu.size()}"
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x_next) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x_next.ndim)

        euler_step = self.euler_step(x_next, d, dt)
        x = self.possible_correction_step(
            euler_step, x_next, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        assert x.size() == denoised.size(), f"Shape mismatch: x {x.size()} != denoised {denoised.size()}"
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
        

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            x, denoised = self.repaint_step(
                i,
                num_sigmas,
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=False
                
            )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x, others



class ResidualHeunEDMSampler(ResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            sigma_bc = append_dims(next_sigma, x.ndim)
            st = self.sigma2st(next_sigma)
            st_derivative = self.sigma2st.get_derivative_st()(next_sigma)
            
            st_bc = append_dims(st, x.ndim)
            st_derivative_bc = append_dims(st_derivative, x.ndim)
            
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, st, uc)
            d_new = (- st_derivative_bc / (st_bc ** 2)) * mu - \
                (denoised + (1 - st_bc) / st_bc * mu - x) / sigma_bc
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x

class TemporalResidualEDMSampler(ResidualEDMSampler):        
    
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False,target=None,mask=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            else:
                x, denoised = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others

class TemporalResidualEulerEDMSampler(TemporalResidualEDMSampler):
    def possible_correction_step(
        self, euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step

class TemporalResidualEulerEDMSampler_Mask(TemporalResidualEulerEDMSampler):
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False,target=None,mask=None,x_mask=None):
        # if x_mask is not None:
        #     # self.x_mask: (b, t, h, w)
        #     # mu: (b, t, c, h, w)
        #     # x:  (b, t, c, h, w)

        #     mask_x = x_mask.float()  # 转为 float 类型
        #     mask_expanded = mask_x.unsqueeze(2)  # → (b, t, 1, h, w)

        #     # 按时相融合：mask=1 → 用 mu；mask=0 → 保留原 x
        #     x = x * (mask_expanded) + mu * (1.0 - mask_expanded)
        # x = x * (1. - new_mask) + target * mask
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if target is not None and mask is not None:
            # 将 mask 转成 float 并确保形状正确
            new_mask = mask.float()
            # print(new_mask.unique())
            if new_mask.shape[1] != 1:  # 如果 mask 是 (b,t,h,w)，先合并成 (b,1,h,w)
                new_mask = (~mask.bool()).any(dim=1, keepdim=True).float()

            # 将 target 映射到相同空间
            new_target = target.clone()

            # 融合：mask=1 → 用 target；mask=0 → 用 denoised
            denoised = denoised * (1. - new_mask) + new_target * new_mask

        if return_attn:
            denoised, attn = denoised
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False,target=None,mask=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        if target is not None and mask is not None:
            # 生成单通道 new_mask：任意时相无云即为1
            new_mask = (~mask.bool()).any(dim=1, keepdim=True).float()  # (b, 1, h, w)
            x_mask = mask.float()
            # 生成 no_cloud 掩膜：1=无云，0=有云
            no_cloud = (mask <0.5).float()   # (b, t, h, w)
            # print(no_cloud.unique())

            # 扩展维度匹配 mu
            no_cloud_expanded = no_cloud.unsqueeze(2)  # (b, t, 1, h, w)

            # 计算无云加权平均
            mu_masked = mu * no_cloud_expanded          # (b, t, c, h, w)
            count_no_cloud = no_cloud_expanded.sum(dim=1, keepdim=False)  # (b, 1, h, w)

            # 防止除零
            has_no_cloud = (count_no_cloud > 0.5).float()
            count_no_cloud = torch.clamp(count_no_cloud, min=0.)

            # 计算平均
            mu_sum = mu_masked.sum(dim=1)               # (b, c, h, w)
            mu_avg = torch.zeros_like(mu_sum)
            mask_expanded = has_no_cloud.expand(-1, mu_sum.shape[1], -1, -1)
            mu_avg[mask_expanded.bool()] = (mu_sum / count_no_cloud)[mask_expanded.bool()]

            # 结果
            target_mapped = mu_avg.float()  # (b, c, h, w)
        else:
            target_mapped = None
            new_mask = None
            x_mask = None
            has_no_cloud = None 
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn,
                    target=target,
                    mask=new_mask,
                    x_mask=x_mask
                )
                
            else:
                x, denoised = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn,
                    target=target,
                    mask=new_mask,
                    x_mask=x_mask
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others



class TemporalResidualEulerEDMSampler_LessMask(TemporalResidualEulerEDMSampler):
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False,target=None,mask=None,x_mask=None):
        # if x_mask is not None:
        #     # self.x_mask: (b, t, h, w)
        #     # mu: (b, t, c, h, w)
        #     # x:  (b, t, c, h, w)

        #     mask_x = x_mask.float()  # 转为 float 类型
        #     mask_expanded = mask_x.unsqueeze(2)  # → (b, t, 1, h, w)

        #     # 按时相融合：mask=1 → 用 mu；mask=0 → 保留原 x
        #     x = x * (mask_expanded) + mu * (1.0 - mask_expanded)
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if target is not None and mask is not None:
            # 将 mask 转成 float 并确保形状正确
            new_mask = mask.float()
            # print(new_mask.unique())
            if new_mask.shape[1] != 1:  # 如果 mask 是 (b,t,h,w)，先合并成 (b,1,h,w)
                new_mask = (~mask.bool()).any(dim=1, keepdim=True).float()

            # 将 target 映射到相同空间
            new_target = target

            # 融合：mask=1 → 用 target；mask=0 → 用 denoised
            denoised = denoised * (1. - new_mask) + new_target * new_mask

        if return_attn:
            denoised, attn = denoised
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False,target=None,mask=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        if target is not None and mask is not None:
            # mask=1 表示有云 → 0 表示无云
            no_cloud = (mask < 0.5).float()  # (b, t, h, w)
            no_cloud_expanded = no_cloud.unsqueeze(2)  # (b, t, 1, h, w)
            x_mask = mask.float()
            # 统计无云时相数
            count_no_cloud = no_cloud_expanded.sum(dim=1, keepdim=False)  # (b, 1, h, w)
            num_t = mask.shape[1]

            # 按比例生成掩膜（[0,1] 区间）
            new_mask = count_no_cloud / num_t   # (b, 1, h, w)

            # 判定“所有时相都有云”的像素：小于一个时相无云
            threshold = 1.0 / num_t
            all_cloud = (count_no_cloud < threshold)

            # 计算 mu 平均
            mu_masked = mu * no_cloud_expanded
            count_no_cloud = torch.clamp(count_no_cloud, min=1e-6)
            mu_sum = mu_masked.sum(dim=1)
            mu_avg = mu_sum / count_no_cloud

            # 对全云区域置零
            new_mask[all_cloud] = 0.0
            mu_avg[all_cloud.expand_as(mu_avg)] = 0.0

            target_mapped = mu_avg.float()
        else:
            target_mapped = None
            new_mask = None
            x_mask = None
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn,
                    target=target,
                    mask=new_mask,
                    x_mask=x_mask
                )
                
            else:
                x, denoised = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn,
                    target=target,
                    mask=new_mask,
                    x_mask=x_mask
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others



class TemporalResidualEulerEDMSampler_Repaint(TemporalResidualEulerEDMSampler):
    def __init__(
        self, repaint_steps = 0, repaint_schurn = 0.0,threshold_diff = 0.01, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.repaint_steps = repaint_steps
        self.repaint_schurn = repaint_schurn
        self.threshold_diff = threshold_diff
    
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step_inner(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
    
    '''
    cv2.imwrite("output.png",(((scale(x0[:,0,])[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha0.png",((((diff)[0][0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha1.png",((((diff)[0][1]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha2.png",((((diff)[0][2]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("attention.png",((((attention_mask)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("unattention.png",((((unattention_mask)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("attention_scaled.png",((((attention_mask_scaled)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("unattention_scaled.png",((((unattention_mask_scaled)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))

    '''
    def repaint_step_inner(self, sigma, next_sigma, denoiser, x,x0, mu, cond, uc=None, gamma=0.0,eps_d = 1e-8):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, False)
        '''
        # TODO: 两张采样步图像组合
        x0:上一个采样步无云结果：主要使用
        denoised:当前采样步无云结果
        
        先统计 x0和denoised差值,选择平均数以上的记为x_temp
        在计算x_temp区域内部,计算x0和mu差值,denoised和mu差值,取差值大的地方mask作为1
        其余mask全部为0
        mask为1的地方用denoised, 0的地方用x0
        '''
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        # _denoised = denoised 
        # print(_denoised.size())
        #print(x0.size())
        # 检查形状是否一致，否则报错
        assert _denoised.size() == x0.size(), f"Shape mismatch: _denoised {_denoised.size()} != x0 {x0.size()}"
        assert _denoised.size() == mu.size(), f"Shape mismatch: _denoised {_denoised.size()} != mu {mu.size()}"
        # 计算x0和denoised的差值（绝对值）

        diff = torch.abs(x0 - _denoised)  # (b, seq, c, h, w)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 对每个通道计算差值的平均值（在后两个维度h,w上取平均）
        channel_means = torch.mean(diff, dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        normalized_diff = (diff - min_val) / (max_val - min_val + eps_d)
        # 创建临时mask（x_temp），标记差值大于平均值的区域
        # x_temp = torch.logical_and(
        #     diff > channel_means,          # 条件1：比均值大
        #     normalized_diff > self.threshold_diff # 条件2：归一化差值 > 阈值
        # ).float()  # 转为 0/1 掩码
        x_temp = (normalized_diff > self.threshold_diff).float()  # 仅保留第二个条件
        temp_mask = (1.0 - x_temp)
        mu_masked = mu * temp_mask
        valid_pixels = temp_mask.sum(dim=(3,4), keepdim=True)
        valid_pixels = torch.clamp(valid_pixels, min=1)  # 防止除以 0
        # 计算x0和mu的差值
        # mu_diff = torch.abs(x0 - mu)  # (b, seq, c, h, w)
        # mu_m =torch.mean(mu, dim=(-2, -1), keepdim=True) 
        mu_m = mu_masked.sum(dim=(3,4), keepdim=True) / valid_pixels
        diff_x0_mu = torch.abs(x0 - mu_m)  # (b, seq, c, h, w)
        # 计算denoised和mu的差值
        diff_denoised_mu = torch.abs(_denoised - mu_m)  # (b, seq, c, h, w)
        
        # 在x_temp区域内比较两个差值
        # 首先创建x_temp区域的掩码
        x_temp_mask = (x_temp > 0)  # bool mask
        
        # 初始化最终mask
        mask = torch.zeros_like(x0)  # (b, seq, c, h, w)
        
        # 在x_temp区域内，比较两个差值，取较大的那个
        # 对于x_temp为1的区域，如果diff_denoised_mu < diff_x0_mu，则mask设为1
        mask[x_temp_mask] = (diff_denoised_mu[x_temp_mask] < diff_x0_mu[x_temp_mask]).float()
        mask *= 1.0
        # 生成最终结果：mask为1的地方用denoised，0的地方用x0
        result = mask * _denoised + (1.0 - mask) * x0
        
        
        return x, result.mean(dim = 1)
        
        
    def get_repaint_allsteps(self):
        return self.repaint_steps 
    
    def repaint_step(self, glb_step,num_sigmas,sigma, next_sigma, denoiser, x, mu, cond, uc=None , gamma =0.0, 
                     return_attn=False):
        
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        re_gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.repaint_schurn / (num_sigmas - 1)
                if self.s_tmin <= sigma.any() <= self.s_tmax
                else 0.0
            )
        if return_attn:
            x, denoised, attn = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        else:
            x, denoised = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        if return_attn:
            denoised, attn = denoised
        base_denoised = denoised
        x_next = x
        # st0 = self.sigma2st(sigmas[0])
        # # x = mu + sigmas[0] * st * x
        # x = ((1 - st0) / st0) * mu + sigmas[0] * x  
        
        repaint_allsteps = self.get_repaint_allsteps()
        assert glb_step <= num_sigmas-2, f"Value mismatch: glb_step : {glb_step} > num_sigmas-2: {num_sigmas-2}"
        if glb_step != 0:
            for i in range(repaint_allsteps):
                # print(glb_step)
                # print(num_sigmas)
                if i == 0:
                    _base_denoised = base_denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                    st0 = self.sigma2st(sigma)
                    # print(st0.size())
                    noise = torch.randn_like(_base_denoised)
                    assert _base_denoised.size() == mu.size(), f"Shape mismatch: _base_denoised {_base_denoised.size()} != mu {mu.size()}"
                    x_r = ((1 - st0) / st0) * mu + _base_denoised + sigma * noise
                else :
                    _denoised_other = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                    st0 = self.sigma2st(sigma)
                    noise = torch.randn_like(_denoised_other)
                    assert _denoised_other.size() == mu.size(), f"Shape mismatch: _denoised_other {_denoised_other.size()} != mu {mu.size()}"
                    x_r = ((1 - st0) / st0) * mu + _denoised_other + sigma * noise
                # if glb_step!=num_sigmas -2:
                x_0 = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                x_prand,new_denoised = self.repaint_step_inner(
                    sigma,
                    next_sigma,
                    denoiser,
                    x_r,
                    x_0,
                    mu,
                    cond,
                    uc,
                    re_gamma,
                )
                
                denoised = new_denoised
                x_next = x_prand
            
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
        assert _denoised.size() == mu.size(), f"Shape mismatch: _denoised {_denoised.size()} != mu {mu.size()}"
        assert x_next.size() == mu.size(), f"Shape mismatch: x_next {x_next.size()} != mu {mu.size()}"
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x_next) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x_next.ndim)

        euler_step = self.euler_step(x_next, d, dt)
        x = self.possible_correction_step(
            euler_step, x_next, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
        

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False,target=None,mask=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.repaint_step(
                    i,
                    num_sigmas,
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            else:
                x, denoised = self.repaint_step(
                    i,
                    num_sigmas,
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others

 
class IdealSampler(SingleStepResidualDiffusionSampler):
    def __init__(self, input_key, mean_key, dataloader_config, *args, **kwargs):
        self.dataloader = instantiate_from_config(dataloader_config)
        self.input_key = input_key
        self.mean_key = mean_key
        super().__init__(*args, **kwargs)
    
    def normal_pdf(self, mean, scale, value):
        assert len(mean.shape) == 4
        assert len(value.shape) == 4 or len(value.shape) == 3
        assert value.shape[0] == 1 or len(value.shape) == 3
        assert mean.shape[-3:] == value.shape[-3:]
        scale_square = scale ** 2
        d = torch.tensor(value.shape).cumprod(dim=-1)[-1]
        return ((2 * torch.pi * scale_square) ** (- d / 2)) * torch.exp(((mean - value) ** 2) / (-2 * scale_square))
    
    def denoise(self, x, sigma):
        divisor = torch.zeros_like(x)
        dividend = torch.zeros_like(x)
        for data in self.dataloader.train_dataloader():
            input = data[self.input_key].to(x.device)
            mu = data[self.mean_key].to(x.device)
            y_i = input + sigma * mu
            # log_prob = torch.distributions.normal.Normal(y_i, sigma).log_prob(x)
            # prob = log_prob.exp()
            prob = self.normal_pdf(y_i, sigma, x)
            divisor += (prob * y_i).sum(dim=0)
            dividend += prob.sum(dim=0)
        return divisor / dividend
    
    def __call__(self, x, mu, num_steps=None, return_intermediate=False, return_denoised=False,target=None,mask=None):
        x = x[0]
        mu = mu[0]
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        st = 1.0 /  (1.0 + sigmas[0])
        x = mu + sigmas[0] * st * x
        
        num_sigmas = len(sigmas)
        range_sigmas = self.get_sigma_gen(num_sigmas)
        # s_in = x.new_ones([x.shape[0]])
        
        intermediates = []
        denoiseds = []
        for i in range_sigmas:
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            st = 1.0 / (1.0 + sigma)
            sigma_bc = append_dims(sigma, x.ndim)
            st_bc = append_dims(st, x.ndim)
            denoised = self.denoise(x / st_bc, sigma_bc)
            
            intermediates.append(x.clone().detach().unsqueeze(0))
            denoiseds.append(denoised.clone().detach().unsqueeze(0))
            
            st = 1.0 / (1.0 + sigma)
            st = append_dims(st, x.ndim)
            d = - st * (x - mu)  - denoised * st / sigma_bc + x / sigma_bc
            dt = append_dims(next_sigma - sigma, x.ndim)
            x = self.euler_step(x, d, dt)
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        return x.unsqueeze(0), others      
    
    
    
    



class TemporalResidualEulerEDMSampler_MAERepaint(TemporalResidualEulerEDMSampler):
    def __init__(
        self, mae_config=None, repaint_steps = 0, repaint_schurn = 0.0,threshold_diff = 0.01, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mae = instantiate_from_config(mae_config)
        self.mae.load_state_dict(torch.load(mae_config['params']['configs']['model_path'], map_location='cpu'), strict=False)
        for param in self.mae.parameters():
            param.requires_grad = False  # 关闭梯度计算
        self.repaint_steps = repaint_steps
        self.repaint_schurn = repaint_schurn
        self.threshold_diff = threshold_diff
        self.scale_01_from_minus1_1 = scale_01_from_minus1_1()
        
    def denoise(self, x, denoiser, sigma, cond, st, uc, return_attn=False):
        if return_attn:
            denoised, attn = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc),st=st, return_attn=return_attn)
        else:
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), st=st, return_attn=return_attn)
        denoised = self.guider(denoised, sigma)
        if return_attn:
            return denoised, attn
        else:
            return denoised
    
    def sampler_step_inner(self, sigma, next_sigma, denoiser, x, mu, cond, uc=None, gamma=0.0, return_attn=False):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, return_attn=return_attn)
        if return_attn:
            denoised, attn = denoised
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
    
    '''
    cv2.imwrite("output.png",(((scale(x0[:,0,])[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha0.png",((((diff)[0][0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha1.png",((((diff)[0][1]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("alpha2.png",((((diff)[0][2]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("attention.png",((((attention_mask)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("unattention.png",((((unattention_mask)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("attention_scaled.png",((((attention_mask_scaled)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))
    cv2.imwrite("unattention_scaled.png",((((unattention_mask_scaled)[0]).transpose(0,1).transpose(1,2).cpu().numpy()*255)).astype(np.uint8))

    '''
    def maerepaint_step_inner(self, sigma, next_sigma, denoiser, x,x0,x_mu, mu, cond, uc=None, gamma=0.0,eps_d = 1e-8):
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + ((1 - st_hat_bc) / st_hat_bc - (1 - st_bc) / st_bc) * mu + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
        
        denoised = self.denoise(x, denoiser, sigma_hat, cond, st_hat, uc, False)
        '''
        # TODO: 两张采样步图像组合
        x0:上一个采样步无云结果：主要使用
        denoised:当前采样步无云结果
        
        先统计 x0和denoised差值,选择平均数以上的记为x_temp
        在计算x_temp区域内部,通过与ref比较得到最后的结果
        其余mask全部为0
        mask为1的地方用denoised, 0的地方用x0
        '''
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        # _denoised = denoised 
        # print(_denoised.size())
        #print(x0.size())
        # 检查形状是否一致，否则报错
        assert _denoised.size() == x0.size(), f"Shape mismatch: _denoised {_denoised.size()} != x0 {x0.size()}"
        assert _denoised.size() == mu.size(), f"Shape mismatch: _denoised {_denoised.size()} != mu {mu.size()}"
        # 计算x0和denoised的差值（绝对值）
        diff = torch.abs(x0 - _denoised)  # (b, seq, c, h, w)
        min_val = diff.amin(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        max_val = diff.amax(dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        # 对每个通道计算差值的平均值（在后两个维度h,w上取平均）
        channel_means = torch.mean(diff, dim=(-2, -1), keepdim=True)  # (b, seq, c, 1, 1)
        normalized_diff = (diff - min_val) / (max_val - min_val + eps_d)
        
        sum_normalized_diff = torch.sum(normalized_diff, dim=(1), keepdim=False)  # (b, 1, h, w)
        # print(sum_normalized_diff.shape)
        sum_normalized_diff = torch.sum(sum_normalized_diff, dim=(1), keepdim=True)  # (b, 1, h, w)
        # print(sum_normalized_diff.shape)
        sum_min = sum_normalized_diff.amin(dim=(-2, -1), keepdim=True)  # (b, 1, 1, 1)
        sum_max = sum_normalized_diff.amax(dim=(-2, -1), keepdim=True)  # (b, 1, 1, 1)
        sum_normalized_diff = (sum_normalized_diff - sum_min) / (sum_max - sum_min + eps_d)  # (b, 1, h, w)

        
        # Create mask (1 where > threshold, 0 otherwise)
        mae_mask = torch.where(sum_normalized_diff > self.threshold_diff, 
                            torch.tensor(0.0, device=normalized_diff.device),
                            torch.tensor(1.0, device=normalized_diff.device))  # (b, 1, h, w)

        # Reshape to (b, h, w, 1) if needed
        mae_mask = mae_mask.permute(0, 2, 3, 1)  # (b, h, w, 1)
        x_mu_mean = x_mu.mean(dim=1,keepdim=False)
        original_ref = ((x_mu_mean+1.0)/2.0).transpose(1, 2).transpose(2, 3)
        original_ref = torch.clamp(original_ref, min=0.0, max=1.0)  # 限制在[0,1]
        self.mae = self.mae.to(original_ref.device)
        original_reference_denoised = self.mae(original_ref,mae_mask)
        # print(original_reference_denoised.shape)
        reference_denoised = original_reference_denoised.transpose(2,3).transpose(1,2)
        reference_denoised = reference_denoised*2.0 -1.0
        reference_denoised = torch.clamp(reference_denoised, min=-1.0, max=1.0)  # 限制在[-1,1
        reference_ =reference_denoised.unsqueeze(dim=1).repeat(1, x.shape[1], 1, 1, 1)
        # print(reference_denoised.shape)
        # 计算与参考的差距
        diff_denoised = torch.abs(_denoised - reference_ )  
        diff_denoised_ = torch.abs(x0 - reference_ )

        x_temp = (normalized_diff > self.threshold_diff).float()  # 仅保留第二个条件
        temp_mask = (1.0 - x_temp)
        
        # 在x_temp区域内比较两个差值
        # 首先创建x_temp区域的掩码
        x_temp_mask = (x_temp > 0)  # bool mask
        
        # 初始化最终mask
        mask = torch.zeros_like(x0)  # (b, seq, c, h, w)
        
        # 在x_temp区域内，比较两个差值，取较小的那个
        # 对于x_temp为1的区域，如果diff_denoised > diff_denoised_ ，则mask设为1
        mask[x_temp_mask] = (diff_denoised[x_temp_mask] < diff_denoised_[x_temp_mask]).float()
        mask *= 1.0
        # 生成最终结果：mask为1的地方用denoised，0的地方用x0
        result = mask * _denoised + (1.0 - mask) * x0
        
        
        return x, result.mean(dim = 1)
        
        
    def get_repaint_allsteps(self):
        return self.repaint_steps 
    
    def repaint_step(self, glb_step,num_sigmas,sigma, next_sigma, denoiser, x, mu, cond, uc=None , gamma =0.0, 
                     return_attn=False):
        #print(sigma)
        st = self.sigma2st(sigma)
        st_bc = append_dims(st, x.ndim)
        sigma_bc = append_dims(sigma, x.ndim)
        sigma_hat = sigma * (gamma + 1.0)
        st_hat = self.sigma2st(sigma_hat)
        st_hat_derivative = self.sigma2st.get_derivative_st()(sigma_hat)
        st_hat_bc = append_dims(st_hat, x.ndim)
        st_hat_derivative_bc = append_dims(st_hat_derivative, x.ndim)
        sigma_hat_bc = append_dims(sigma_hat, x.ndim)
        re_gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.repaint_schurn / (num_sigmas - 1)
                if self.s_tmin <= sigma.any() <= self.s_tmax
                else 0.0
            )
        if return_attn:
            x, denoised, attn = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        else:
            x, denoised = self.sampler_step_inner(
                sigma,
                next_sigma,
                denoiser,
                x,
                mu,
                cond,
                uc,
                gamma,
                return_attn=return_attn
            )
        if return_attn:
            denoised, attn = denoised
        base_denoised = denoised
        x_next = x
        # st0 = self.sigma2st(sigmas[0])
        # # x = mu + sigmas[0] * st * x
        # x = ((1 - st0) / st0) * mu + sigmas[0] * x  
        
        repaint_allsteps = self.get_repaint_allsteps()
        assert glb_step <= num_sigmas-2, f"Value mismatch: glb_step : {glb_step} > num_sigmas-2: {num_sigmas-2}"
        if glb_step != 0:
            for i in range(repaint_allsteps):
                # print(glb_step)
                # print(num_sigmas)
                if i == 0:
                    _base_denoised = base_denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                    st0 = self.sigma2st(sigma)
                    st0_bc = append_dims(st, x.ndim)
                    sigma_bc = append_dims(sigma, x.ndim)
                    # print(st0.size())
                    noise = torch.randn_like(_base_denoised)
                    assert _base_denoised.size() == mu.size(), f"Shape mismatch: _base_denoised {_base_denoised.size()} != mu {mu.size()}"
                    x_r = ((1 - st0_bc) / st0_bc) * mu + _base_denoised + sigma_bc * noise
                    x_mu = ((1 - st0_bc) / st0_bc) * mu + _base_denoised
                else :
                    _denoised_other = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                    st0 = self.sigma2st(sigma)
                    st0_bc = append_dims(st, x.ndim)
                    sigma_bc = append_dims(sigma, x.ndim)
                    noise = torch.randn_like(_denoised_other)
                    assert _denoised_other.size() == mu.size(), f"Shape mismatch: _denoised_other {_denoised_other.size()} != mu {mu.size()}"
                    x_r = ((1 - st0_bc) / st0_bc) * mu + _denoised_other + sigma_bc * noise
                    x_mu = ((1 - st0_bc) / st0_bc) * mu + _denoised_other
                # if glb_step!=num_sigmas -2:
                x_0 = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
                x_prand,new_denoised = self.maerepaint_step_inner(
                    sigma,
                    next_sigma,
                    denoiser,
                    x_r,
                    x_0,
                    x_mu,
                    mu,
                    cond,
                    uc,
                    re_gamma,
                )
                
                denoised = new_denoised
                x_next = x_prand
            
        _denoised = denoised.unsqueeze(dim=1).repeat(1, x_next.shape[1], 1, 1, 1)
        assert _denoised.size() == mu.size(), f"Shape mismatch: _denoised {_denoised.size()} != mu {mu.size()}"
        assert x_next.size() == mu.size(), f"Shape mismatch: x_next {x_next.size()} != mu {mu.size()}"
        d = (- st_hat_derivative_bc / (st_hat_bc ** 2)) * mu - \
            (_denoised + (1 - st_hat_bc) / st_hat_bc * mu - x_next) / sigma_hat_bc
        dt = append_dims(next_sigma - sigma_hat, x_next.ndim)

        euler_step = self.euler_step(x_next, d, dt)
        x = self.possible_correction_step(
            euler_step, x_next, mu, d, dt, next_sigma, denoiser, cond, uc
        )
        if return_attn:
            return x, denoised, attn
        else:
            return x, denoised
        

    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, return_intermediate=False, return_denoised=False, return_attn=False,target=None,mask=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, mu, cond, uc, num_steps
        )
        intermediates = []
        denoiseds = []
        attns = []
        range_sigmas = self.get_sigma_gen(num_sigmas)
        for i in range_sigmas:
            gamma = (
                # min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                self.s_churn / (num_sigmas - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            if return_intermediate:
                intermediates.append(tools_scale(x.clone().detach()))
            if return_attn:
                x, denoised, attn = self.repaint_step(
                    i,
                    num_sigmas,
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            else:
                x, denoised = self.repaint_step(
                    i,
                    num_sigmas,
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x,
                    mu,
                    cond,
                    uc,
                    gamma,
                    return_attn=return_attn
                )
            if return_denoised:
                denoiseds.append(tools_scale(denoised.detach()))
            if return_attn:
                # attns.append(tools_scale(attn.detach()))
                attns.append(attn.detach())
        others = {}
        if return_intermediate:
            others["intermediates"] = intermediates
        if return_denoised:
            others["denoiseds"] = denoiseds
        if return_attn:
            others["attns"] = attns
        return x.mean(dim=1), others
    


