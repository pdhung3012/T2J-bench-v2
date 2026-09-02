import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @vmap
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Softmax activation to the input array.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, num_features).

        Returns:
            jnp.ndarray: Output array with Softmax applied, same shape as input.
        """
        return jnp.softmax(x, axis=1)

batch_size = 4096
dim = 393216

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
