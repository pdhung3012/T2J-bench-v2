import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Softmax, Initializer

class Model:
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        cumsum = jnp.cumsum(x.at[Ellipsis, :self.dim].reduce(jnp.add, 0), axis=self.dim)
        return jnp.concatenate((jnp.zeros_like(x[:, :1]), cumsum), axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size,) + input_shape)]

def get_init_inputs():
    return [dim]
