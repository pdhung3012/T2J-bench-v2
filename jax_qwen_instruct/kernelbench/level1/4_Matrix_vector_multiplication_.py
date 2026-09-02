import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return jnp.matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

get_inputs = jit(vmap(lambda: (jnp.random.rand(M, K), jnp.random.rand(K, 1))))

get_init_inputs = lambda: []  # No special initialization inputs needed
