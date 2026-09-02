import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha
    
    @jit
    def foward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies ELU activation to the input array.

        Args:
            x (jnp.ndarray): Input array of any shape.

        Returns:
            jnp.ndarray: Output array with ELU applied, same shape as input.
        """
        return jnp.where(x >= 0, x, self.alpha * (jnp.exp(x) - 1))

batch_size = 4096
dim = 393216

get_inputs = jit(vmap(lambda: jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))))

get_init_inputs = lambda: [1.0]  # Provide alpha value for initialization
