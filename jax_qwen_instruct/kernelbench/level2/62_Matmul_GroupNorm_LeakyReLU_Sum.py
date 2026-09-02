import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, GroupNorm, LeakyRelu

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = Dense(hidden_size)
        self.gn = GroupNorm(num_groups=num_groups, num_channels=hidden_size, epsilon=eps)
        self.leaky_relu = LeakyRelu(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

get_inputs = lambda: [jnp.random.rand(batch_size, input_size).astype(jnp.float32)]
get_init_inputs = lambda: [input_size, hidden_size, num_groups]
