import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input array with shape (M, K).
            B: Input array with shape (K, N).

        Returns:
            C: Output array with shape (M, N).
        """
        return jnp.matmul(A, B)

M = 8205
K = 2949
N = 5921

get_inputs = jit(vmap(lambda: (jnp.random.rand(M, K), jnp.random.rand(K, N))))

get_init_inputs = lambda: []  # No special initialization inputs needed
