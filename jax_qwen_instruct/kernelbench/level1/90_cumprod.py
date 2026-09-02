import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Grad, OptInit, Initializer, ParamInit, Params, BatchNorm, Flatten

class CumulativeProductModel:
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        self.dim = dim

    def __call__(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, *input_shape).

        Returns:
            jnp.ndarray: Array of the same shape as `x` after applying cumulative product along `dim`.
        """
        return jnp.cumprod(x, axis=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, *input_shape))]

def get_init_inputs():
    return [dim]
