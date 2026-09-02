import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after Max reduction over the specified dimension.
        """
        return jnp.max(x, axis=self.dim)[0]

batch_size = 128
dim1 = 4096
dim2 = 4095

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim1, dim2)))

get_init_inputs = lambda: [1] # Example, change to desired dimension
