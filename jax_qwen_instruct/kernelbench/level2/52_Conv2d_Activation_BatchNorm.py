import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Tanh, SoftmaxPlus

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
        self.bn = BatchNorm(eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = jnp.multiply(jnp.tanh(jnp.nn.softplus(x)), x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
