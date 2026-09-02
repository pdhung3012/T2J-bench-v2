import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    @vmap
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, input_size).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        original_x = x
        x = jnp.sigmoid(x)
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size))]

get_init_inputs = lambda: [input_size, hidden_size, scaling_factor]
