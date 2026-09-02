import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(42)
batch_size = 1
seq_len = 3
dim = 3

q = random.normal(key, (batch_size, seq_len, dim))
k = random.normal(key, (batch_size, seq_len, dim))
v = random.normal(key, (batch_size, seq_len, dim))

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute the scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output tensor of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights tensor of shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]  # Get the last dimension size (key dimension)
    
    # Compute the dot product of Q and K^T
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(jnp.array(d_k, dtype=jnp.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights along the last dimension
    attention_weights = jax.nn.softmax(scores, axis=-1)  # dim=-1 ensures softmax is applied across the last axis
    
    # Compute output by weighting V with the attention weights
    output = jnp.matmul(attention_weights, v)
    
    return output, attention_weights

# Testing on data & compare
output_custom, _ = scaled_dot_product_attention(q, k, v)
print(output_custom)
output = F.scaled_dot_product_attention(q, k, v)
print(output)

assert jnp.allclose(output_custom, output, atol=1e-08, rtol=1e-05)
