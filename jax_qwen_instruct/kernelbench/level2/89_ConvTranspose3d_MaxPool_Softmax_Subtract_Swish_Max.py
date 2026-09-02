import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int
    pool_kernel_size: int
    pool_stride: int
    pool_padding: int

    @nn.compact
    def __init__(self, *, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding,
                 _shared_params=None):
        super().__init__(**dict(_shared_params))
        self.conv_transpose = nn.ConvTranspose3D(features=out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
                                                 strides=(stride, stride, stride), padding='SAME', output_padding=(output_padding, output_padding, output_padding))
        self.max_pool = nn.MaxPool3D(window_shape=(pool_kernel_size, pool_kernel_size, pool_kernel_size), strides=(pool_stride, pool_stride, pool_stride),
                                     padding='SAME')
        self.subtract = nn.Parameter(jnp.zeros(out_channels), name='subtract')

    def __call__(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = jnp.softmax(x, axis=-1)  # Apply softmax across channels (axis=-1 for last dimension)
        x = x - self.subtract  # Subtract across channels
        x = jnp.where(x > 0, jnp.sigmoid(x) * x, 0)  # Swish activation
        x = jnp.max(x, axis=-1)  # Max pooling across channels
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, depth, height, width).astype(jnp.float32)]
get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]
