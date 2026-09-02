import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (jnp.ndarray): Input tensor of arbitrary shape.

        Returns:
            jnp.ndarray: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return jnp.mean(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim1, dim2)))

get_init_inputs = lambda: [1]
