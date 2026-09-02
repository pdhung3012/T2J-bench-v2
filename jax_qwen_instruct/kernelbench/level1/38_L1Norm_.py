import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(Model, self).__init__()

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies L1 normalization to the input array.

        Args:
            x (jnp.ndarray): Input array of shape (..., dim, ...).

        Returns:
            jnp.ndarray: Output array with L1 normalization applied, same shape as input.
        """
        return x / jnp.expand_dims(jnp.mean(jnp.abs(x), axis=1), axis=-1)

batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))
    return [x]

def get_init_inputs():
    return []
