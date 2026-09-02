import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(Model, self).__init__()
        self.dim = dim

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Array containing the indices of the minimum values along the specified dimension.
        """
        return jnp.argmin(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim1, dim2), in_axes=(None,), out_axes=0))

get_init_inputs = lambda: [dim]
