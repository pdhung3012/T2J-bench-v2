import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = jnp.min(x, axis=1, keepdims=True)  # Apply minimum operation along the channel dimension
        x = jnp.tanh(x)
        x = jnp.tanh(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

get_inputs = jax.jit(vmap(get_inputs))

get_init_inputs = jax.jit(vmap(get_init_inputs))
