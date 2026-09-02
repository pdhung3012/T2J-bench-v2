import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): Input matrix A of shape (N, N).
            B (jnp.ndarray): Input matrix B of shape (N, N).

        Returns:
            jnp.ndarray: Output matrix C of shape (N, N).
        """
        return jnp.matmul(A, B)

N = 2048 * 2

@jit
def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N, N))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, N))
    return [A, B]

@jit
def get_init_inputs():
    return []  # No special initialization inputs needed
