import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (jnp.ndarray): Input 3D tensor of shape (N, M, K).
            B (jnp.ndarray): Input matrix of shape (K, L).

        Returns:
            jnp.ndarray: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return jnp.matmul(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N, M, K))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(K, L))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
