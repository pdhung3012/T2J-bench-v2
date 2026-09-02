import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Relu, HardSigmoid

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')

    def forward(self, x):
        x = self.conv(x)
        x = Relu(x)
        x = x * jnp.clip((x + 3) / 6, 0, 1)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
