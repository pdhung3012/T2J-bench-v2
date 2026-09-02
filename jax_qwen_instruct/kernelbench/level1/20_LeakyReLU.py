import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies LeakyReLU activation to the input array.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with LeakyReLU applied, same shape as input.
        """
        return jnp.where(x < 0, x * self.negative_slope, x)

batch_size = 4096
dim = 393216

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))]

get_init_inputs = lambda: []  # No special initialization inputs needed
