import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor with argmax applied, with the specified dimension removed.
        """
        return jnp.argmax(x, axis=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, dim1, dim2)))

get_init_inputs = lambda: [1]
