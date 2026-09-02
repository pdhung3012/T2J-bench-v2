import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @vmap
    def swish(self, x):
        return x * jnp.sigmoid(x)
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Swish activation to the input array.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with Swish applied, same shape as input.
        """
        return self.swish(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = jnp.random.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
