from abc import abstractmethod

import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sgm.modules.diffusionmodules.k_diffusion import flags, flops, layers


try:
    import natten
except ImportError:
    natten = None

try:
    import flash_attn
except ImportError:
    flash_attn = None

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x) 

class GateGELU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * F.gelu(x2)


class OgGateGeLU(nn.Module):
    def __init__(self, features,dims=2):
        super().__init__()
        self.Enlarge = conv_nd(dims,features,features*2,1,padding = 0)
        self.activation = GateGELU()
        self.skip_activation = GELU()
        
    def forward(self, x):
        skip =x
        x = self.Enlarge(x)
        x = self.activation(x)
        skip = self.skip_activation(skip)
        return skip + x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

    
    

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(16, channels)
    # return LayerNorm2d(channels)


class AdaptModulation(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),  # 平滑非线性激活
            zero_init(nn.Linear(cond_features, 2 * features, bias=True))
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x):
        return self.modulation(x)  # (B, 2C)



class AdaptLayerNorm(nn.Module):
    def __init__(self, sizes, cond_features):
        super().__init__()
        self.norm = nn.LayerNorm(sizes, elementwise_affine=False)  # 禁用默认的 affine 参数
        self.adaLN_modulation = AdaptModulation(sizes[0], cond_features)
    
    
    def forward(self,x , cond):
        c = self.adaLN_modulation(cond)[:, :,None, None]
        scale, shift = c.chunk(2,dim=1)
        x_norm = self.norm(x) 
        
        return x_norm*(1+scale)+shift
    
    
        

def normalization_temporal(sizes,cond_features):
    
    return AdaptLayerNorm(sizes,cond_features)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """




class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

_initial_missing = object()
def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, iterable[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """

    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value") from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value
def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))
        # tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


class TimestepEmbedlevel(nn.ModuleList, TimestepBlock):

    def __init__(self, modules=None):
        super().__init__(modules)

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class TemporalLevel(nn.ModuleList, TimestepBlock):
    def __init__(self, modules=None):
        super().__init__(modules)
    
    def smart_forward(self, layer, input,cond, *args, **kwargs):
        if len(input.shape) == 4:
            if isinstance(layer, TimestepBlock):
                return layer(input, cond, *args, **kwargs)
            else:
                return layer(input, *args, **kwargs)
        else:
            b, t, c, h, w = input.shape

            out = input.contiguous().view(b * t, c, h, w)
            b, seq_L,channels = cond.shape
            out_cond = cond.contiguous().view(b *  seq_L, channels)
            if isinstance(layer, TimestepBlock):
                out = layer(out, out_cond, *args, **kwargs)
            else:
                
                out = layer(out, *args, **kwargs)
            _, c, h, w = out.shape
            out = out.contiguous().view(b, t, c, h, w)
            return out
    
    def forward(self, x, cond, *args, **kwargs):
        for layer in self:
            x = self.smart_forward(layer, x, cond, *args, **kwargs)
        return x
           

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_channel_down_activation = False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if use_channel_down_activation ==False:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.GELU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            self.updown = up or down
            if up:
                self.h_upd = Upsample(channels, False, dims)
                self.x_upd = Upsample(channels, False, dims)
            elif down:
                self.h_upd = Downsample(channels, False, dims)
                self.x_upd = Downsample(channels, False, dims)
            else:
                self.h_upd = self.x_upd = nn.Identity()

            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.out_channels,
                ),
            )
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                OgGateGeLU(self.out_channels,dims),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            elif use_conv:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, 3, padding=1
                )
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.GELU(),
                conv_nd(dims, channels, channels, 3, padding=1),
            )
            self.updown = up or down

            if up:
                self.h_upd = Upsample(channels, False, dims)
                self.x_upd = Upsample(channels, False, dims)
            elif down:
                self.h_upd = Downsample(channels, False, dims)
                self.x_upd = Downsample(channels, False, dims)
            else:
                self.h_upd = self.x_upd = nn.Identity()

            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    channels,
                ),
            )
            self.out_layers = nn.Sequential(
                normalization(channels),
                GateGELU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, channels//2,self.out_channels, 3, padding=1)
                ),
            )

            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            elif use_conv:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, 3, padding=1
                )
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

       

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale= emb_out
            tmp = out_norm(h)
            h =  tmp * (1 + scale) 
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ResBlock_Cond(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_channel_down_activation = False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if use_channel_down_activation == False:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.GELU(),
                conv_nd(dims, channels, self.out_channels, 3, padding=1),
            )
            self.updown = up or down
            if up:
                self.h_upd = Upsample(channels, False, dims)
                self.x_upd = Upsample(channels, False, dims)
            elif down:
                self.h_upd = Downsample(channels, False, dims)
                self.x_upd = Downsample(channels, False, dims)
            else:
                self.h_upd = self.x_upd = nn.Identity()

            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                OgGateGeLU(self.out_channels,dims),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            elif use_conv:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, 3, padding=1
                )
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            self.in_layers = nn.Sequential(
                normalization(channels),
                nn.GELU(),
                conv_nd(dims, channels, channels, 3, padding=1),
            )
            self.updown = up or down

            if up:
                self.h_upd = Upsample(channels, False, dims)
                self.x_upd = Upsample(channels, False, dims)
            elif down:
                self.h_upd = Downsample(channels, False, dims)
                self.x_upd = Downsample(channels, False, dims)
            else:
                self.h_upd = self.x_upd = nn.Identity()

            self.out_layers = nn.Sequential(
                normalization(channels),
                GateGELU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, channels//2,self.out_channels, 3, padding=1)
                ),
            )

            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            elif use_conv:
                self.skip_connection = conv_nd(
                    dims, channels, self.out_channels, 3, padding=1
                )
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).contiguous().view(bs * self.n_heads, ch, length),
            (k * scale).contiguous().view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class LTAELayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_head=16,
        d_k=4,
        mlp_conv=[256, 512],
        dropout=0.2,
        dims=2,
        sequence_len =3,
    ):
        class MultiHeadAttention(nn.Module):
            def __init__(self, n_head, d_k, d_in,dropout=0.0,dims=2,sequence_len = 3):
                super().__init__()
                self.d_k = d_k
                self.d_in = d_in
                self.d_head = d_in // n_head
                self.n_head = n_head
                self.norm = normalization(d_in)
                self.fc1_q = conv_nd(1,sequence_len,1,1,padding = 0)
                self.fc1_qk = conv_nd(1,d_in, d_k * n_head*2,1,padding = 0)
                self.dropout = nn.Dropout(dropout)
                self.out_proj = zero_init(conv_nd(dims,d_in, d_in,1,padding =0))
                # self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
                # nn.init.normal_(self.Q, mean=0, std=(2.0 / (d_k)) ** 0.5)

            def forward(self, v):
                d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
                sz_b, h, w ,seq_len, d = v.shape
                v = v.contiguous().view(sz_b, d, seq_len, -1)
                v = self.norm(v)
                v = v.contiguous().view(sz_b * h * w, seq_len, d)
                # k = self.fc1_k(v.contiguous().view(sz_b * h * w, d,seq_len)).view(sz_b * h * w, seq_len, n_head, d_k)
                # k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
                # q = torch.stack([self.Q for _ in range(sz_b * h * w)], dim=1).view(-1, d_k)
                qk =self.fc1_qk(v.contiguous().view(sz_b * h * w, d,seq_len))
                q,k = qk.split(d_k*n_head, dim=1)
                k = k.view(sz_b * h * w, seq_len, n_head, d_k)
                k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
                q = self.fc1_q(q.permute(0,2,1).contiguous()).view(-1,d_k)
                v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1), dim=0).view(n_head * sz_b * h * w, seq_len, -1)
                attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / (d_k ** 0.5)
                attn = F.softmax(attn, dim=-1)
                output = torch.matmul(attn, v)
                attn = attn.view(n_head, sz_b * h * w, 1, seq_len)
                attn = attn.squeeze(dim=2)
                output = output.view(n_head, sz_b * h * w, 1, d_in // n_head)
                output = output.squeeze(dim=2)
                output = output.permute(1, 0, 2).contiguous().view(sz_b * h * w, d_in)
                output = self.dropout(output)
                output = output.contiguous().view(sz_b ,d_in, h ,w)
                output = self.out_proj(output)
                return output, k,v
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert mlp_conv[0] == d_model
        self.mlp_conv = mlp_conv
        self.positional_encoder = None
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model, dropout=dropout,dims = dims,sequence_len=sequence_len
        )
        self.ff = nn.Sequential(
            normalization(d_model),
            conv_nd(dims,self.d_model,self.mlp_conv[1],1,padding=0),
            GateGELU(),
            nn.Dropout(dropout),
            conv_nd(dims,self.d_model,self.d_model,1,padding = 0)
        )
        

    def forward(self, x):
        sz_b, seq_len, d, h, w = x.shape
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        out, k_cond,v_cond = self.attention_heads(out.view(sz_b, h, w, seq_len, d))
        skip = out
        out = self.ff(out) +skip
        # attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
        #     0, 1, 4, 2, 3
        # )
        return out, k_cond,v_cond


class Temporal_CrossAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in,d_cond_in,dropout=0.0,dims=2,sequence_len = 3):
        super().__init__() 
        self.d_k = d_k
        self.d_in = d_in
        self.d_head = d_in // n_head
        self.n_head = n_head
        self.d_cond_in=d_cond_in
        self.dropout = nn.Dropout(dropout)
        self.norm = normalization(d_in)
        self.fc2_q = conv_nd(1,sequence_len,1,1,padding = 0)
        self.fc1_q = conv_nd(1,d_in, d_k * n_head,1,padding = 0)
        self.out_proj = zero_init(conv_nd(dims,d_cond_in, d_in,1,padding =0))
        
    def forward(self,x,k,v):
        d_k, d_in, n_head,d_cond_in = self.d_k, self.d_in, self.n_head, self.d_cond_in
        sz_b, h, w ,seq_len, d = x.shape
        x = x.contiguous().view(sz_b, d, seq_len, -1)
        x = self.norm(x)
        x = x.contiguous().view(sz_b * h * w, seq_len, d)
        q =self.fc1_q(x.contiguous().view(sz_b * h * w, d,seq_len))
        q = self.fc2_q(q.permute(0,2,1).contiguous()).view(-1,d_k)
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / (d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        output = output.view(n_head, sz_b * h * w, 1, d_cond_in // n_head)
        output = output.squeeze(dim=2)
        output = output.permute(1, 0, 2).contiguous().view(sz_b * h * w, d_cond_in)
        output = self.dropout(output)
        output = output.contiguous().view(sz_b ,d_cond_in, h ,w)
        output = self.out_proj(output)
        return output
        


class Patchize(nn.Module):
    def __init__(self,img_size = 256, dims = 2,in_features=64, out_features=64, patch_size=[4, 4],pos_emb = True, x_dims = 4):
        super().__init__()
        self.out_shape = None
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.pos_emb = pos_emb
        self.out_features = out_features
        self.x_dims = x_dims
        # self.H = (img_size // patch_size[0])
        # self.W =(img_size // patch_size[1])
        self.num_patches = (img_size // patch_size[0]) * (img_size // patch_size[1]) 
        self.proj = conv_nd(dims, in_features * self.h * self.w, out_features, 3, padding=1)
        if self.pos_emb:
            if x_dims == 4:
                self.position_embed = nn.Parameter(torch.zeros(1, out_features,img_size//patch_size[0],img_size//patch_size[1]))
            elif x_dims ==5:
                self.position_embed = nn.Parameter(torch.zeros(1, 1, out_features,img_size//patch_size[0],img_size//patch_size[1]))
 

    def forward(self, x):
        # print(x.shape)
        x = rearrange(x, "... e (h nh) (w nw)-> ... (nh nw e) h w", nh=self.h, nw=self.w)
        if self.x_dims ==4:
            if self.pos_emb:
                return self.proj(x)+self.position_embed
            else:
                return self.proj(x) 
        elif self.x_dims == 5:
            sz_b,seq_len, d ,h,w= x.shape
            if self.pos_emb:
                out = self.proj(x.view(sz_b*seq_len,d,h,w))+self.position_embed
            else:
                out = self.proj(x.view(sz_b*seq_len,d,h,w))
            return out.contiguous().view(sz_b,seq_len,self.out_features,h,w) 
        # print(x.shape)
        # print(self.position_embed.shape)
        


class UnPatchize(nn.Module):
    def __init__(self, dims=2, in_features=64, out_features=64, patch_size=[4, 4],x_dims = 4):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.x_dims = x_dims
        self.out_channels =out_features * self.h * self.w 
        self.proj = conv_nd(dims, in_features, self.out_channels, 3,padding = 1)
    
    def forward(self, x):
        # x = self.proj(x)
        if self.x_dims ==4:
            x = self.proj(x)
        elif self.x_dims == 5:
            x = self.proj(x)
        return rearrange(x, "... (nh nw e) h w -> ... e (h nh) (w nw)", nh=self.h, nw=self.w)

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)

def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)

class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class Temporal_FusionBlock(nn.Module):
    def __init__(
        self,
        img_size,
        in_features = 7,
        mid_channels = 64,
        out_features = 7,
        dims = 2,
        n_head = 1,
        dropout = 0.0,
        d_k = 64,
        d_cond_in = 64,
        sequence_len = 3,
        size_ctimes = 3,
        patch_mult = 1,
        time_emb = 64,
        x_dims = 5
        ):
        super().__init__()
        self.img_size = img_size
        self.in_features = in_features
        self.mid_channels = mid_channels
        self.out_features = out_features
        self.n_head = n_head
        self.d_cond_in = d_cond_in
        self.dropout = dropout
        self.d_k = d_k
        self.sequence_len = sequence_len
        self.size_ctimes = size_ctimes
        self.hw = img_size
        self.patch_hw = 1
        self.x_dims = x_dims
        for i in range(size_ctimes):
            self.hw = self.hw // 2
            self.patch_hw *= 2
        self.patch_size = [self.patch_hw, self.patch_hw]
        patch_channels =self.mid_channels * patch_mult
        self.patchify = Patchize(img_size=img_size,dims = dims, in_features= self.in_features,out_features=patch_channels,patch_size=self.patch_size,pos_emb=True,x_dims=x_dims)
        self.cross_attention = Temporal_CrossAttention(n_head=n_head,d_k=d_k,d_in = patch_channels,
                                                       d_cond_in=d_cond_in,dropout=dropout,
                                                       dims = dims, sequence_len = sequence_len)
        self.unpatchify = UnPatchize(dims=dims,in_features=patch_channels,out_features=self.out_features,patch_size=self.patch_size)
        self.adptnorm = normalization_temporal([out_features,img_size,img_size], cond_features=time_emb) 
        self.ff = nn.Sequential(
            OgGateGeLU(out_features,dims),
            nn.Dropout(dropout),
            conv_nd(dims,out_features,out_features,1,padding = 0)
        )
        
    def forward(self,x,k_cond,v_cond,emb):
        x = self.patchify(x)
        sz_b,seq_len, d ,h,w= x.shape
        out = self.cross_attention(x.contiguous().view(sz_b,h,w,seq_len,d),k_cond,v_cond)
        # print(out.shape)
        out = self.unpatchify(out)
        skip = out
        out = self.adptnorm(out,emb)
        out = self.ff(out)
        
        return out + skip
    


class GlobalTransformer(nn.Module):
    
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Local_Perception(nn.Module):
    def __init__(
        self,
        dims = 2,
        channels=64,
        use_checkpoint=False,
    ):
        super().__init__() 
        self.dims = 2
        self.in_channels = channels
        self.out_channels = channels
        self.local_perception =nn.Sequential(
            normalization(self.in_channels),
            nn.GELU(),
            conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1),
            normalization(self.out_channels),
        )
    
    def forward(self, x):
        skip_x =x
        x = self.local_perception (x) +skip_x
        return x


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = list(kernel_size)
        self.norm = normalization(d_model)
        self.qkv_proj = conv_nd(2,d_model, d_model * 3, 3,padding = 1)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        # self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(conv_nd(2,d_model, d_model, 3,padding = 1))

    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        # print(self.kernel_size)
        if natten.has_fused_na():
            # 分割QKV并重排为多头形式
            
            q, k, v = rearrange(qkv, 'b (three nh dh) h w -> three b h w nh dh', 
                            three=3, nh=self.n_heads, dh=self.d_head)
            # 使用scale_for_cosine_sim进行缩放（保留原始逻辑）
            # print(f"输入形状检查 -> q: {q.shape}, k: {k.shape}, scale: {self.scale.shape}")
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
            # print(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
            # theta = self.pos_emb(pos)
            # q = apply_rotary_emb_(q, theta)
            # k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0)
            x = rearrange(x, "n h w nh e -> n (nh e) h w")
        else:
            q, k, v = rearrange(qkv, "n (t nh e) h w -> t n nh h w e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None, None], 1e-6)
            # theta = self.pos_emb(pos).movedim(-2, -4)
            # q = apply_rotary_emb_(q, theta)
            # k = apply_rotary_emb_(k, theta)
            flops.op(flops.op_natten, q.shape, k.shape, v.shape, self.kernel_size)
            qk = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = torch.softmax(qk, dim=-1).to(v.dtype)
            x = natten.functional.na2d_av(a, v, self.kernel_size)
            x = rearrange(x, "n nh h w e -> n (nh e) h w")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


'''
# TODO: 混合注意力修改：neigborhood attention + 不要分patch
'''
class Mixed_Preception(TimestepBlock):
    def __init__(
        self,
        dims = 2,
        img_size = 256,
        kernel_size = 3,
        emb_channels = 64,
        channels=64,
        num_heads=1,
        num_head_channels=-1,
        neighbor_heads = 64,
        use_checkpoint=False,
        use_new_attention_order=False,
        pos_emb = True,
    ):
        super().__init__() 
        self.dims = 2
        self.double_layer =self.local_perception =nn.Sequential(
            normalization(channels),
            conv_nd(dims, channels, 2*channels, 1, padding=0)
        )
        channels =channels *2
        self.emb_activation = nn.SiLU()
        # self.patch_size = patch_size
        # self.in_channels = channels//2
        self.out_channels = channels //2
        # self.local_perception =nn.Sequential(
        #     normalization(self.in_channels),
        #     nn.GELU(),
        #     conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1),
        #     normalization(self.out_channels)
        # )
        self.local_perception = NeighborhoodSelfAttentionBlock(self.out_channels, neighbor_heads,
                                                               kernel_size,dropout=0.0)
        self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels ,
                ),
            )
        
        # patch_channels = self.out_channels * patch_mult
        # self.patchify = Patchize(img_size=img_size,dims = dims, in_features= self.in_channels,out_features=patch_channels,patch_size=patch_size,pos_emb=pos_emb)
        # self.attention_layernorm = normalization(self.out_channels)
        # self.out_features = patch_channels
        self.globalattention = GlobalTransformer(
            channels = self.out_channels, use_checkpoint=use_checkpoint,
            num_heads=num_heads,num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order
        )
        # self.unpatchify = UnPatchize(dims=dims,in_features=patch_channels,out_features=self.out_channels,patch_size=patch_size)
        self.final_proj = nn.Sequential(normalization(self.out_channels),
                                        nn.GELU(),
                                        zero_init(conv_nd(dims,self.out_channels, self.out_channels,1,padding =0)),
                                     )
    
    def forward(self, x,emb):
        skip_x = x
        h = self.double_layer(x)
        x1, x2 = h.chunk(2, dim=1)
        emb_out = self.emb_layers(emb).type(x1.dtype)
        while len(emb_out.shape) < len(x1.shape):
            emb_out = emb_out[..., None]
        emb_out = self.emb_activation(emb_out)
        scale1, scale2 = th.chunk(emb_out, 2, dim=1)
        # emb_stacked = torch.stack([1. +scale1, 1.+ scale2], dim=-1)  # 形状: (batch_size, C // 2, H, W, 2)
        # # 对最后一个维度应用 softmax
        # emb_softmaxed = torch.softmax(emb_stacked, dim=-1)  # 形状: (batch_size, C // 2, H, W, 2)
        # # 拆分 softmax 后的结果
        # scale1_norm, scale2_norm = emb_softmaxed.unbind(dim=-1)  # 形状: (batch_size, C // 2, H, W)
        x1 = self.local_perception (x1)
        # x2 = self.attention_layernorm(x2)
        x2 = self.globalattention(x2)
        x_att = x1*(1+scale1)+x2*(1+scale2)
        x=self.final_proj(x_att) + skip_x
        return x 


class Condition_UNet(nn.Module):
    """
    L个block处理多时序图像,中间加上t作为embedding
    融合diffcr进行condition输入,y作为整体输入
    
    """

    def __init__(
        self,
        image_size=256,
        in_channels=4,
        main_in_channels = 7,
        model_channels = 64,  # even number
        condition_net_multiple = 2,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions = [32,16,8],
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        seq_length = 3,
        d_k = 64,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        time_emb_fourier = True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.seq_length = seq_length
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.d_k = d_k
        
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        self.time_emb_fourier = time_emb_fourier
        
        self.time_emb = FourierFeatures(1, self.model_channels)
        self.time_in_proj = nn.Linear(self.model_channels, self.model_channels, bias=False)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_preview = TemporalLevel([conv_nd(dims, in_channels, ch, 3, padding=1)])
        self.input_blocks = nn.ModuleList(
            #[TemporalLevel([conv_nd(dims, in_channels, ch, 3, padding=1)])]
        )
        input_block_chans = []
        ds = 1
        for level, mult in enumerate(channel_mult):
            ch_home =int(mult * model_channels) 
            ch_cin = ch_home
            for _ in range(num_res_blocks):
                if _ == 0:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch_home,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch_cin = ch
                    ch = ch_home
                else:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            # use_channel_down_activation = True,
                        )
                    ]
                    ch_cin =ch
                    ch = ch
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TemporalLevel(layers))
                input_block_chans.append(ch_cin*condition_net_multiple)
            
            if level != len(channel_mult) - 1:
                out_ch = ch
                ch_cin = ch
                self.input_blocks.append(
                    TemporalLevel([
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_channel_down_activation=False,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    ])
                )
                ch = out_ch
                ds *= 2
                input_block_chans.append(ch_cin*condition_net_multiple)
        
        input_block_chans.reverse()
        cond_channel_count = []
        
        self.middle_preview = LTAELayer(
            d_model = ch,
            n_head = self.num_heads,
            d_k = self.d_k,
            mlp_conv = [ch,ch*2],
            dropout= self.dropout,
            dims = dims,
            sequence_len = self.seq_length,
        )
        self.cond_mid_ch = ch
        # concat original L sequence
        self.middle_block = TimestepEmbedlevel([
            # ResBlock(
            #     ch*self.seq_length,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
            # TemporalLevel([conv_nd(dims, ch*self.seq_length, ch, 1, padding=0)]),
            
            
            
            ResBlock_Cond(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                out_channels=ch,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(
            #     ch//2,
            #     use_checkpoint=use_checkpoint,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            # ResBlock_Cond(
            #     ch//2,
            #     time_embed_dim,
            #     dropout,
            #     dims=dims,
            #     out_channels=ch,
            #     use_checkpoint=use_checkpoint,
            #     use_scale_shift_norm=use_scale_shift_norm,
            # ),
        ])

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            ch_up_home =int(model_channels * mult) 
            for i in range(num_res_blocks):
                # if i == 0:
                #     ich = input_block_chans.pop()
                # else:   
                if i == 0:
                    ich = 0
                    layers = [
                        ResBlock_Cond(
                            ch + ich*self.seq_length,
                            #ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = int(model_channels * mult)
                elif i ==num_res_blocks:
                    ich = 0
                    layers = [
                        ResBlock_Cond(
                            ch + ich*self.seq_length,
                            #ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                else:
                    ich = 0
                    layers = [
                        ResBlock_Cond(
                            ch + ich*self.seq_length,
                            #ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            # use_channel_down_activation = True
                        )
                    ] 
                    ch = ch
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.output_blocks.append(TimestepEmbedlevel(layers))
                cond_channel_count.append(ch)
            if level:
                out_ch = ch
                layers = []
                layers.append(
                    ResBlock_Cond(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=True,
                    )
                    if resblock_updown
                    else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                ds //= 2
                
                self.output_blocks.append(TimestepEmbedlevel(layers))
                cond_channel_count.append(ch)
        self.cond_channel_count = cond_channel_count.copy()
        self.out = nn.ModuleList([])
        assert len(input_block_chans) == len(self.cond_channel_count)
        for k, m_ch in enumerate(self.cond_channel_count):
            connect_out_chan = input_block_chans[k]
            self.out.append(nn.Sequential(
                normalization(m_ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, m_ch, connect_out_chan, 3, padding=1)),
            ))
        
        # self.out = nn.Sequential(
        #     normalization(ch),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        # )
    
    '''
      设想是cond处理之后返回对应h*w的图像,与主干网络进行concat,因此需要提前知道返回的
      channel和h&w数值,用于定义主干网络的1*1卷积核输入深度
    '''
    def get_condition_channel(self):
          
        return self.cond_channel_count
    
    '''
    返回中间attention的key和value的channel
    '''
    def get_condmid_channel(self):
        
        return self.cond_mid_ch

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, dates=None, return_attn=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(x.shape)
        b, t , c, h, w = x.shape
        # x = x.contiguous().view(b, t * c, h, w) 
        timesteps = torch.arange(t).repeat(b, 1).to(x.device)
        
        # hs = []
        if self.time_emb_fourier:  # fourier embedding choice
            emb = self.time_embed(self.time_in_proj(self.time_emb(timesteps[..., None])))
        else:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        mid_cond_output = []
        # hs = []
        h = x.type(self.dtype)
        h = self.input_preview(h,emb)
        for k,module in enumerate(self.input_blocks):
            h = module(h, emb)
            # if (k+1) % (self.num_res_blocks+1)==0:
            #     hs.append(h)
            # hs.append(h)
        skip_mh = h.mean(dim=1)
        h , k_cond, v_cond = self.middle_preview(h)
        h = h+skip_mh
        
        # b0, t0 , c0, h0, w0 = h.shape
        # h = h.contiguous().view(b0, t0*c0, h0, w0)  # make L*channels
        
        h = self.middle_block(h, emb)
        for k,module in enumerate(self.output_blocks):
            #hs_tmp = hs.pop()
            #b1, t1 , c1, h1, w1 = hs_tmp.shape
            #hs_tmp = hs_tmp.contiguous().view(b1, t1*c1, h1, w1)  # make L*channels
            #h = th.cat([h, hs_tmp], dim=1)
            # if (k+1)%(self.num_res_blocks+1)==1:
            #     hs_tmp = hs.pop() 
            #     b1, t1 , c1, h1, w1 = hs_tmp.shape
            #     hs_tmp = hs_tmp.contiguous().view(b1, t1*c1, h1, w1)  # make L*channels
            #     h = th.cat([h, hs_tmp], dim=1) 
            h = module(h, emb)
            mid_out = self.out[k](h)
            mid_out = mid_out.type(x.dtype)
            mid_cond_output.append(mid_out)
        # mid_cond_output.reverse()
        # h = h.type(x.dtype)
        return mid_cond_output,k_cond,v_cond


# TODO: 降低计算量

class UNetModel(nn.Module):
    """
    L个block处理多时序图像，中间加上t作为embedding
    融合diffcr进行condition输入，y作为整体输入
    
    
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size=256,
        in_channels=7,
        model_channels = 64,  # even number
        noise_channels = 3,
        condition_net_multiple = 2,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions = [32,16,8],
        sequence_length = 3,
        d_k = 64,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        attention_names=['local','local','local','local'],
        patch_size = [4,4],
        kernel_size = [3,3],
        neighbor_heads = 64,
        patch_mult = 1,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        time_emb_fourier = True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.sequence_length = sequence_length
        self.d_k = d_k
        self.dropout = dropout
        self.dims = dims
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.neighbor_heads = neighbor_heads
        self.num_heads_upsample = num_heads_upsample
        self.patch_mult = patch_mult
        self.time_emb_fourier = time_emb_fourier
        img_size = self.image_size
        self.time_emb = FourierFeatures(1, self.model_channels)
        self.time_in_proj = nn.Linear(self.model_channels, self.model_channels, bias=False)
        self.kernel_size = kernel_size
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.attention_names = attention_names
        self.noise_channels = noise_channels
        self.condition_in_channels = in_channels - noise_channels
        self.sequence_length = sequence_length
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        # 初始化条件网络
        self.condition_network = Condition_UNet(image_size= self.image_size,in_channels = self.condition_in_channels,main_in_channels=self.in_channels,
                                                model_channels = self.model_channels//condition_net_multiple,condition_net_multiple=condition_net_multiple,out_channels = self.out_channels,
                                                num_res_blocks = self.num_res_blocks,attention_resolutions = self.attention_resolutions,
                                                dropout = self.dropout,channel_mult = self.channel_mult,conv_resample = self.conv_resample,
                                                dims = self.dims,seq_length = self.sequence_length,d_k = self.d_k, use_checkpoint = False,use_fp16=False, num_heads=self.num_heads,num_head_channels = self.num_head_channels,
                                                 num_heads_upsample =self.num_heads_upsample,use_scale_shift_norm = self.use_scale_shift_norm,
                                                resblock_updown =self.resblock_updown,use_new_attention_order =self.use_new_attention_order,
                                                time_emb_fourier =self.time_emb_fourier
                                                )
        #获取条件网络输出相关信息
        self.cond_channel_count = self.condition_network.get_condition_channel()
        cond_channel_count = self.cond_channel_count.copy()
        
        self.d_cond_in = self.condition_network.get_condmid_channel()
        
        self.temporal_fusion = Temporal_FusionBlock(
            img_size= self.image_size, in_features= self.in_channels,mid_channels=self.d_k,
            out_features=self.in_channels,dims=dims, n_head =self.num_heads,dropout =self.dropout,
            d_k = self.d_k, d_cond_in = self.d_cond_in, sequence_len = self.sequence_length,size_ctimes= len(channel_mult)-1,
            patch_mult=self.patch_mult,time_emb=time_embed_dim,x_dims = 5
        )
        
        ch = input_ch = int(channel_mult[0] * model_channels)
        # ch_cond = cond_channel_count.pop()
        self.cond_add_ch = self.out_channels
        
        # self.input_temporal_fusion = TimestepEmbedlevel([conv_nd(dims, in_channels*self.sequence_length, in_channels, 3, padding=1)])
        self.input_priview =TimestepEmbedlevel([conv_nd(dims, in_channels, ch, 3, padding=1)])
        self.input_blocks = nn.ModuleList(
            # [TimestepEmbedlevel([conv_nd(dims, in_channels, ch, 3, padding=1)])]
        )
        input_block_chans = []
        ds = 1
        for level, mult in enumerate(channel_mult):
            perception_name = attention_names[level]
            ch_home = int(mult * model_channels)
            for _ in range(num_res_blocks):
                if _ == 0:
                    ch_cond = cond_channel_count.pop()
                    layers = [
                        # conv_nd(dims, ch+ch_cond, ch, 1, padding=0),
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch_home,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = ch_home 
                else:
                    ch_cond = cond_channel_count.pop()
                    layers = [
                        # conv_nd(dims, ch+ch_cond, ch, 1, padding=0),
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            # use_channel_down_activation =True
                        )
                    ]
                    ch = ch
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if _ == num_res_blocks-1 and level == len(channel_mult) - 1:
                    layers.append(
                        nn.Identity()
                        if perception_name == 'local'
                        else Mixed_Preception(dims=dims,img_size =img_size,channels = ch,
                                              num_heads = num_heads,
                                              emb_channels = time_embed_dim,
                                              kernel_size = self.kernel_size,
                                              num_head_channels = num_head_channels,neighbor_heads = self.neighbor_heads,
                                              use_checkpoint=use_checkpoint,
                                              use_new_attention_order=use_new_attention_order)
                    )
                    ch = ch
                
                self.input_blocks.append(TimestepEmbedlevel(layers))
                    
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                ch_cond = cond_channel_count.pop()
                img_size=img_size //2
                self.input_blocks.append(
                    TimestepEmbedlevel([
                        # conv_nd(dims, ch+ch_cond, ch, 1, padding=0),
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        ),
                        nn.Identity()
                        if perception_name == 'local'
                        else Mixed_Preception(dims=dims,img_size =img_size,channels = out_ch,
                                              num_heads = num_heads,
                                              emb_channels = time_embed_dim,
                                              kernel_size = self.kernel_size,
                                              num_head_channels = num_head_channels,neighbor_heads=self.neighbor_heads,
                                              use_checkpoint=use_checkpoint,
                                              use_new_attention_order=use_new_attention_order)
                    ])
                )
                #ch = out_ch//2
                ch = out_ch
                ds *= 2

                

        self.middle_block = TimestepEmbedlevel([
            Mixed_Preception(dims=dims,img_size =img_size,channels = ch,
                                              num_heads = num_heads,
                                              emb_channels = time_embed_dim,
                                              kernel_size = self.kernel_size,
                                              num_head_channels = num_head_channels,neighbor_heads = self.neighbor_heads,
                                              use_checkpoint=use_checkpoint,
                                              use_new_attention_order=use_new_attention_order,pos_emb=False),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels =ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ])

        self.output_blocks = nn.ModuleList([])
        attention_names.reverse()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            ch_home =int(model_channels * mult)
            perception_name = attention_names[level]
            for i in range(num_res_blocks):
                if i ==0 :
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=ch_home,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                    ch = ch_home
                else:
                    ich = 0
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            # use_channel_down_activation = True
                        )
                    ]
                    ch = ch
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if i == num_res_blocks:
                    out_ch = ch
                    layers.append(    
                        nn.Identity() 
                        if perception_name == 'local'
                        else 
                        Mixed_Preception(dims=dims,img_size =img_size,channels = out_ch,
                                              num_heads = num_heads,
                                              emb_channels = time_embed_dim,
                                              kernel_size = self.kernel_size,
                                              num_head_channels = num_head_channels,neighbor_heads = self.neighbor_heads,
                                              use_checkpoint=use_checkpoint,
                                              use_new_attention_order=use_new_attention_order)
                    )
                    ch = out_ch
                self.output_blocks.append(TimestepEmbedlevel(layers))
            if level:
                out_ch = ch
                layers = []
                layers.append(
                    # ResBlock(
                    #     out_ch//2,
                    #     time_embed_dim,
                    #     dropout,
                    #     out_channels=out_ch//2,
                    #     dims=dims,
                    #     use_checkpoint=use_checkpoint,
                    #     use_scale_shift_norm=use_scale_shift_norm,
                    #     up=True,
                    # )
                    # if resblock_updown
                    # else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    ResBlock(
                        out_ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=True,
                    )
                    if resblock_updown
                    else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    
                )
                img_size = int(img_size*2)
                ds //= 2
                #ch = out_ch//2
                ch = out_ch
                self.output_blocks.append(TimestepEmbedlevel(layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps,  dates=None, return_attn=False,y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        # print(x.shape)
        if x.dim() == 4:  # 如果 x 是 4 个维度  
            b, c, h, w = x.shape   
            x = x.unsqueeze(1) 
        b, t , c, h, w = x.shape
        # condition = x.clone().contiguous().view(b, t , c, h, w)
        condition = x[:, :, self.noise_channels:, :, :] 
        # x = x.contiguous().view(b, t * c, h, w) 
        condition_results,k_cond,v_cond = self.condition_network(condition) # 获取所有条件结果
        x = x.contiguous().view(b, t , c, h, w) 
        skip_xt = x.mean(dim=1)
        hs = []
        if self.time_emb_fourier:  # fourier embedding choice
            emb = self.time_embed(self.time_in_proj(self.time_emb(timesteps[..., None])))
        else:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        h = self.temporal_fusion(h,k_cond,v_cond,emb) + skip_xt
        h = self.input_priview(h,emb)
        for k,module in enumerate(self.input_blocks):
            cond_temporal = condition_results.pop()
            #print(cond_temporal.shape,h.shape)
            #h = torch.cat((h, cond_temporal), dim=1)
            h = h + cond_temporal
            h = module(h, emb)
            if (k+1)%(self.num_res_blocks+1)==self.num_res_blocks:
                hs.append(h)
        h = self.middle_block(h, emb)
        for k,module in enumerate(self.output_blocks):
            if (k+1)%(self.num_res_blocks+1)==1:
                h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


