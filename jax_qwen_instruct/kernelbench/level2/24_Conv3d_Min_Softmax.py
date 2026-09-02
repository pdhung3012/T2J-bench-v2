import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import softmax

class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    @vmap
    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, D, H, W)
        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = jnp.min(x, axis=self.dim, keepdims=True)  # Apply minimum along the specified dimension
        x = softmax(x, axis=1)  # Apply softmax along the channel dimension
        return x

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, D, H, W))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, dim]
