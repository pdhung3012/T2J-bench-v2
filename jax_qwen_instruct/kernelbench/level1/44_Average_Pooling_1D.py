import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    @vmap
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            jnp.ndarray: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, input_length))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
