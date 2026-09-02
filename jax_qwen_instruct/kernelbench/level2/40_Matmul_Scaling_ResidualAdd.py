import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_features).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        original_x = x.copy()
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, scaling_factor]
