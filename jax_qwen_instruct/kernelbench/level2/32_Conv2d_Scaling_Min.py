import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    @vmap
    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, height, width).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = x * self.scale_factor
        x = jnp.minimum(x, axis=1, keepdims=True)  # Minimum along channel dimension
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, scale_factor]
