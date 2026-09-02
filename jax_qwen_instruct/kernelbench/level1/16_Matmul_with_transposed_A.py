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
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return jnp.matmul(A.T, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

get_inputs = jit(vmap(lambda: (jax.random.normal(key=jax.random.PRNGKey(0), shape=(K, M)), 
                               jax.random.normal(key=jax.random.PRNGKey(1), shape=(K, N)))))

get_init_inputs = lambda: []  # No special initialization inputs needed
