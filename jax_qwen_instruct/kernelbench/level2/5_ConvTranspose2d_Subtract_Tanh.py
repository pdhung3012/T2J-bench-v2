import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    bias_shape: tuple

    def setup(self):
        self.conv_transpose = nn.ConvTranspose2D(self.in_channels, self.out_channels, self.kernel_size, strides=(2, 2), padding='SAME')
        self.bias = nn.Parameter(jnp.zeros(self.bias_shape))

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = x - self.bias
        x = jnp.tanh(x)
        return x

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, height, width)]
get_init_inputs = lambda: [in_channels, out_channels, kernel_size, bias_shape]
