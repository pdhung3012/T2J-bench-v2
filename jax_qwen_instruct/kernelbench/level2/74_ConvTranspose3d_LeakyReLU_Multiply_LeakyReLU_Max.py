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
    multiplier_shape: tuple

    def setup(self):
        self.conv_transpose = nn.ConvTranspose3D(features=self.out_channels, 
                                                  output_shape=(None, self.out_channels, self.depth // self.stride, self.height // self.stride, self.width // self.stride), 
                                                  kernel_size=self.kernel_size, 
                                                  strides=(self.stride,), 
                                                  paddings=((self.padding, self.padding),) * 3)
        self.multiplier = nn.Parameter(jnp.ones(self.multiplier_shape))
        self.leaky_relu = nn.functional.leaky_relu
        self.max_pool = nn.MaxPool3D(pool_size=2)

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, depth, height, width)]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
