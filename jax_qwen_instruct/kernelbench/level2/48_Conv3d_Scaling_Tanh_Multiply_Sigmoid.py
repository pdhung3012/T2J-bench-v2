import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    scaling_factor: jnp.ndarray
    bias_shape: tuple

    def setup(self):
        self.conv = nn.Conv(self.in_channels, self.out_channels, self.kernel_size)
        self.scaling_factor = nn.Parameter(self.bias_shape)
        self.bias = nn.Parameter(self.bias_shape) 

    @nn.compact
    def __call__(self, x):
        x = self.conv(x)
        x = x * self.scaling_factor 
        x = jnp.tanh(x)
        x = x * self.bias
        x = jnp.sigmoid(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

get_inputs = lambda: [jax.random.normal(next_rng_keys()[0], (batch_size, in_channels, depth, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]
