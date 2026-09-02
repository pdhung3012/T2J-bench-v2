import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @jit
    def foward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies HardTanh activation to the input array.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with HardTanh applied, same shape as input.
        """
        return jnp.clip(x, -1., 1.)

batch_size = 4096
dim = 393216

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
