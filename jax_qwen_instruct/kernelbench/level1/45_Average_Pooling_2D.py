import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            jnp.ndarray: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(x)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, channels, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size]
