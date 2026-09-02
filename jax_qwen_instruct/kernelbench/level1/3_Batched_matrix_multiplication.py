import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @vmap
    def forward(self, A, B):
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return jnp.matmul(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, m, k))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(batch_size, k, n))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
