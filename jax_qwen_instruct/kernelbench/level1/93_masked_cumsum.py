import jax
import jax.numpy as jnp

class Model:
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    @jax.jit
    def forward(self, x, mask):
        return jnp.cumsum(x * mask, axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, *input_shape))
    mask = jax.random.randint(jax.random.PRNGKey(1), x.shape).astype(jnp.bool_)  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]
