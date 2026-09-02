import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs matrix multiplication.

        Args:
            A: Input array of shape (M, K).
            B: Input array of shape (K, N).

        Returns:
            Output array of shape (M, N).
        """
        return jnp.matmul(A, jnp.transpose(B))

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(M, K))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, K))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
