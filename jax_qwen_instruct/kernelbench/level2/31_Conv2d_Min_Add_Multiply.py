import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, Min, Add, Scale

class Model:
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        self.params = [
            Conv(in_channels, out_channels, kernel_size),
            Dense(bias_shape[0]),
            Min(),
            Add(),
            Scale(scaling_factor)
        ]
        self.constant_value = constant_value

    @vmap
    def forward(self, x, params):
        x = jnp.stack([f(x) for f, _ in params])
        x = jnp.min(x, axis=0, initial=self.constant_value)
        x = x + params[1](x)
        x = x * params[-1](x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
