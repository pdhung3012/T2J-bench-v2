import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (jnp.ndarray): Lower triangular matrix of shape (M, M).
            B (jnp.ndarray): Lower triangular matrix of shape (M, M).

        Returns:
            jnp.ndarray: The result of matrix multiplication C of shape (M, M).
        """
        return jnp.tril(jnp.matmul(A, B))

M = 4096

@jit
def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(M, M))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(M, M))
    A = jnp.tril(A)
    B = jnp.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
