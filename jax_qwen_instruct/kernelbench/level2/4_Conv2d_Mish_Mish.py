import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
import haiku as hk

class Model(hk.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = hk.Conv2D(out_channels, (kernel_size, kernel_size), padding='SAME')

    @vmap
    def forward(self, x):
        x = self.conv(x)
        x = jnp.nn.mish(x)
        x = jnp.nn.mish(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
