import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, LeakyRelu, Gelu, Init

class Model(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
        self.multiplier = Dense(out_channels, use_bias=False)
        self.leaky_relu = LeakyRelu()
        self.gelu = Gelu()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier(x)
        x = self.leaky_relu(x)
        x = self.gelu(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 64
height, width = 256, 256
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs(seed=jax.random.PRNGKey(0)):
    key, subkey = random.split(seed)
    return [random.normal(subkey, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]
