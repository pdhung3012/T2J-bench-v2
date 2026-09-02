import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (jnp.ndarray): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            jnp.ndarray: Output matrix of shape (M, N) or (N, M)
        """
        return jnp.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(M, N))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
