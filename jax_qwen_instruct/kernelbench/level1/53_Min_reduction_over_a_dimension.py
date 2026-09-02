import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    @partial(jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after min reduction over the specified dimension.
        """
        return jnp.min(x, axis=self.dim)[0]

batch_size = 128
dim1 = 4096
dim2 = 4095

get_inputs = jit(vmap(lambda: jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim1, dim2))))

get_init_inputs = lambda: [1] # Example, change to desired dimension
