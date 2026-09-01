import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
# Synthetic data
torch.manual_seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = torch.rand(batch_size, seq_len, d_model)
k = torch.rand(batch_size, seq_len, d_model)
v = torch.rand(batch_size, seq_len, d_model)
print(q.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
# Apply RoPE to real query/key tensors.
# Rotary(x) returns the (cos, sin) tables for the sequence length of x,
# and apply_rotary_pos_emb rotates q and k with them.
max_seq_len = 100
d_model = 64
seq_len = 50
batch_size = 2

q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)

custom_pos_emb = Rotary(d_model)
cos, sin = custom_pos_emb(q, seq_dim=1)

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

print(q_rot.shape, k_rot.shape)  # (2, 50, 64) (2, 50, 64)

# A rotation preserves vector norms - a quick sanity check.
print("norms preserved:",
      torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4))