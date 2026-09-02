import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np

# Synthetic data
np.random.seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = jnp.random.rand(batch_size, seq_len, d_model)
k = jnp.random.rand(batch_size, seq_len, d_model)
v = jnp.random.rand(batch_size, seq_len, d_model)
print(q.shape)

device = "cuda" if jax.devices("cuda") else "cpu"
device = "cpu"

class Rotary(jax.namedtuple_tracer_class_type):
    def __init__(self, dim, base=10000):
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
        self.inv_freq = inv_freq
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    @jit
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = jnp.arange(x.shape[seq_dim], device=x.device).astype(jnp.float32)
            freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:

@jit
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=x1.ndim - 1)

@jit
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Apply RoPE to real query/key tensors.
# Rotary(x) returns the (cos, sin) tables for the sequence length of x,
# and apply_rotary_pos_emb rotates q and k with them.
max_seq_len = 100
d_model = 64
seq_len = 50
batch_size = 2

q = jnp.randn(batch_size, seq_len, d_model)
k = jnp.randn(batch_size, seq_len, d_model)

custom_pos_emb = Rotary(d_model)
cos, sin = custom_pos_emb(q, seq_dim=1)

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

print(q_rot.shape, k_rot.shape)  # (2, 50, 64) (2, 50, 64)

# A rotation preserves vector norms - a quick sanity check.
print("norms preserved:",
      jnp.allclose(q_rot.norm(axis=-1), q.norm(axis=-1), atol=1e-4))
