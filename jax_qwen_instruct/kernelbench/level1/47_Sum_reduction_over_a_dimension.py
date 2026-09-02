import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (jnp.ndarray): Input array of shape (..., dim, ...).

        Returns:
            jnp.ndarray: Output array after sum reduction, shape (..., 1, ...).
        """
        return jnp.sum(x, axis=self.dim, keepdims=True)

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim1, dim2)))

get_init_inputs = lambda: [reduce_dim]
