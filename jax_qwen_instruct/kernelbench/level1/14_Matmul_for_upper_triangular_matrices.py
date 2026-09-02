import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (jnp.ndarray): Upper triangular matrix of shape (N, N).
            B (jnp.ndarray): Upper triangular matrix of shape (N, N).

        Returns:
            jnp.ndarray: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return jnp.triu(jnp.matmul(A, B))

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = jnp.triu(jax.random.normal(key=jax.random.PRNGKey(0), shape=(N, N)))
    B = jnp.triu(jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, N)))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []
