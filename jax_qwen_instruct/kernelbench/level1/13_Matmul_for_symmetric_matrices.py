import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (jnp.ndarray): Input matrix A, shape (N, N), symmetric.
            B (jnp.ndarray): Input matrix B, shape (N, N), symmetric.

        Returns:
            jnp.ndarray: Output matrix C, shape (N, N).
        """
        return jnp.matmul(A, B)

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = jnp.random.rand(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = jnp.random.rand(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []
