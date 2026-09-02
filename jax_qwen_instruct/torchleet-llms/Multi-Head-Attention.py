import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

# Synthetic data
key = random.PRNGKey(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = random.normal(key, (batch_size, seq_len, d_model))
k = random.normal(key, (batch_size, seq_len, d_model))
v = random.normal(key, (batch_size, seq_len, d_model))

print(q.shape)

device = "cuda" if jax.device_count() > 0 else "cpu"
device = device

class MultiHeadAttention(nn.Module):
    @nn.compact
    def __call__(self, q, k, v, *, key_axes=None, value_axes=None, batch_first=True):
        batch_size = jax.lax.shape_padlen(q)
        q = q.reshape(batch_size, -1, num_heads, d_model // num_heads).transpose((0, 2, 1, 3))
        k = k.reshape(batch_size, -1, num_heads, d_model // num_heads).transpose((0, 2, 1, 3))
        v = v.reshape(batch_size, -1, num_heads, d_model // num_heads).transpose((0, 2, 1, 3))

        q *= d_model**-0.5
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', q, k)
        
        if mask is not None:
            attn_weights += mask
        
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        output = output.transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, d_model)
        
        return output

key = random.PRNGKey(42)
model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output_custom = model(q, k, v)
print(output_custom)

multihead_attn = nn.MultiheadAttention(d_model=d_model, num_heads=num_heads, bias=False, batch_first=True)
output, _ = multihead_attn(q, k, v)
print(output)

assert jnp.allclose(output_custom, output, atol=1e-08, rtol=1e-05) # Check if they are close enough.
