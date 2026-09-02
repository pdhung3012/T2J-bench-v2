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
    bias_shape: tuple

    def setup(self):
        self.conv_transpose = nn.ConvTranspose2D(self.in_channels, self.out_channels, 
                                                  self.kernel_size, self.stride, self.padding, 
                                                  self.output_padding)
        self.bias = nn.Parameter(jnp.zeros(self.bias_shape))

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = jnp.minimum(x, axis=1, keepdims=True)  # Minimum operation along channel dimension
        x = jnp.sum(x, axis=2, keepdims=True)  # Sum operation along height dimension
        x = jax.nn.gelu(x)  # GELU activation
        x = x + self.bias
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, height, width)]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
