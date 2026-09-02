import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim
    
    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies LogSoftmax activation to the input array.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, dim).

        Returns:
            jnp.ndarray: Output array with LogSoftmax applied, same shape as input.
        """
        return jnp.log(jnp.softmax(x, axis=self.dim))

batch_size = 4096
dim = 393216

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim)))

get_init_inputs = lambda: []  # No special initialization inputs needed
