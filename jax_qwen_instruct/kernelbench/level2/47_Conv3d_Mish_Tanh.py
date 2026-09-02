import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Mish, Tanh, Dense, BatchNorm, Flatten, Relu

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding)
        self.bn = BatchNorm()
        self.relu = Relu()
        self.mish = Mish()
        self.tanh = Tanh()

    @vmap
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mish(x)
        x = self.tanh(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
