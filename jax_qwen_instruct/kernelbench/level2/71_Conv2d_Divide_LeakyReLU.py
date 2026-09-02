import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, LeakyRelu, MaxPool, Flatten, Initializer

class Model(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), stride=1, padding='SAME')
        self.divisor = divisor

    @nn.compact
    def __call__(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = LeakyRelu(alpha=0.01)(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
