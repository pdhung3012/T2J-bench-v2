import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    @jax.jit
    def forward(self, Q: jnp.ndarray, K: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
        out = jax.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    key = random.PRNGKey(0)
    Q = random.normal(key, (batch_size, num_heads, sequence_length, embedding_dimension))
    K = random.normal(key, (batch_size, num_heads, sequence_length, embedding_dimension))
    V = random.normal(key, (batch_size, num_heads, sequence_length, embedding_dimension))
    return [Q, K, V]

def get_init_inputs():
    return []
