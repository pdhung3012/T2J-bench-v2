import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def _ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(jnp.log(10 * jnp.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Dense(features=512, kernel_init=jax.nn.initializers.normal(stddev=0.02), bias_init=jax.nn.initializers.zeros)
        self.cpb_mlp.add_module('activation', nn.gelu)
        self.cpb_mlp.add_module('linear2', nn.Dense(features=num_heads, kernel_init=jax.nn.initializers.zeros))

        self.relative_coords_table = jnp.stack(jnp.meshgrid([-self.window_size[0] + 1, -self.window_size[1] + 1], [-self.window_size[0] + 1, -self.window_size[1] + 1])).T.reshape(-1, 2).astype(jnp.float32)
        if pretrained_window_size[0] > 0:
            self.relative_coords_table /= (pretrained_window_size[0] - 1)
            self.relative_coords_table[:, 1] /= (pretrained_window_size[1] - 1)
        else:
            self.relative_coords_table /= (self.window_size[0] - 1)
            self.relative_coords_table[:, 1] /= (self.window_size[1] - 1)
        self.relative_coords_table *= 8
        self.relative_coords_table = jnp.sign(self.relative_coords_table) * jnp.log2(jnp.abs(self.relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_position_index", jnp.zeros((self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1]), dtype=jnp.int32))

        self.qkv = nn.Dense(features=3 * dim, kernel_init=jax.nn.initializers.normal(stddev=0.02), bias_init=jax.nn.initializers.zeros)
        if qkv_bias:
            self.q_bias = nn.Parameter(jnp.zeros(dim), requires_grad=True)
            self.v_bias = nn.Parameter(jnp.zeros(dim), requires_grad=True)
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(features=dim, kernel_init=jax.nn.initializers.normal(stddev=0.02), bias_init=jax.nn.initializers.zeros)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def _get_relative_position_index(self, window_h, window_w):
        coords_h = jnp.arange(window_h)
        coords_w = jnp.arange(window_w)
        coords_h, coords_w = jnp.meshgrid(coords_h, coords_w)
        coords_flatten = jnp.stack((coords_h.ravel(), coords_w.ravel()), axis=-1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose((1, 2, 0))
        relative_coords[:, :, 0] += window_h - 1
        relative_coords[:, :, 1] += window_w - 1
        relative_coords[:, :, 0] *= 2 * window_w - 1
        return relative_coords.sum(-1)

    def _get_attn_mask(self, mask, h, w):
        if mask is None:
            return None
        nW = mask.shape[0]
        mask = mask.view(nW, h // self.window_size, w // self.window_size, 2)
        mask_windows = window_partition(mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 2)
        mask_windows = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask_windows = mask_windows.masked_fill(mask_windows != 0, -100.0)
        mask_windows = mask_windows.masked_fill(mask_windows == 0, 0.0)
        return mask_windows

    def __call__(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = jnp.concatenate((self.q_bias, jnp.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (jnp.linalg.norm(q, axis=-1)**2 + jnp.linalg.norm(k, axis=-1)**2 - jnp.einsum('bhid,bhjd->bhij', q, k)) / (jnp.sqrt(C) * 2)
        logit_scale = jnp.clip(self.logit_scale, max=jnp.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.transpose((2, 0, 1)).reshape(-1, self.num_heads)
        relative_position_bias = 16 * jnp.tanh(relative_position_bias)
        attn = attn + relative_position_bias[:, None, :]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Dropout(rate=drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = jnp.zeros((1, H, W, 1))  # 1 H W 1
