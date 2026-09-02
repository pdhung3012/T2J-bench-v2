import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def forward(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return jnp.matmul(A, B)

M = 256
N = 256
K = 131072 * 4

get_inputs = jit(vmap(lambda: (jax.random.normal(key=jax.random.PRNGKey(0), shape=(M, K)), 
                               jax.random.normal(key=jax.random.PRNGKey(1), shape=(K, N)))))

get_init_inputs = lambda: []  # No special initialization inputs needed
