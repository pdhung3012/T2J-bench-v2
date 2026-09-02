import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @vmap
    def gelu(self, x):
        return 0.5 * x * (1 + jnp.tanh(
            0.7978845608 * (x + 0.044715 * jnp.power(x, 3))
        ))

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
