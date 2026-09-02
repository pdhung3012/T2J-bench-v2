import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs average pooling, 3D transposed convolution, clamping,
    spatial softmax, and multiplication by a learnable scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.avg_pool = jax.nn.avg_pool
        self.conv_transpose = jax.nn.conv_transpose
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = jax.nn.initializers.ones((1, out_channels, 1, 1, 1))

    @vmap
    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, in_channels, depth, height, width).

        Returns:
            jax.numpy.ndarray: Output array of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x, window_shape=self.pool_kernel_size, strides=self.stride, padding=self.padding)
        x = self.conv_transpose(x, output_shape=x.shape[0], output_size=(x.shape[1],), kernel_shape=self.kernel_size, strides=self.stride, padding=self.padding, output_padding=self.output_padding)
        x = jnp.clip(x, self.clamp_min, self.clamp_max)
        b, c, d, h, w = x.shape
        x = x.reshape(b, c, -1)                     # flatten spatial dims
        x = jax.nn.softmax(x, axis=2)
        x = x.reshape(b, c, d, h, w)
        x = x * self.scale
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, depth, height, width)]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]
