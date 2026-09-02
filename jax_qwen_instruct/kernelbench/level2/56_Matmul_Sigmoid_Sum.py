import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)
        x = jnp.sigmoid(x)
        x = jnp.sum(x, axis=1, keepdims=True)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

get_inputs = jax.jit(vmap(get_inputs))

get_init_inputs = jax.jit(vmap(get_init_inputs))
