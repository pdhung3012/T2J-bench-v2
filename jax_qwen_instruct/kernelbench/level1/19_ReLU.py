import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @vmap
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies ReLU activation to the input array.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with ReLU applied, same shape as input.
        """
        return jnp.maximum(0, x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
