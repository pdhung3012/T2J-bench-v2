import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Flatten, MaxPool, Conv, BatchNorm, ScaleShift, Stacks, Serial, InputLayer

class Model:
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        self.dim = dim

    @jax.jit
    def fprop(self, x):
        return jnp.flip(jnp.cumsum(jnp.flip(x, axis=self.dim)), axis=self.dim)

def get_inputs():
    return [jnp.random.rand(*input_shape).astype(jnp.float32)]

def get_init_inputs():
    return [dim]
