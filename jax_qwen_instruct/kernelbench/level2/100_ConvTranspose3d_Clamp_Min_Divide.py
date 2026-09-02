import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, Flatten, Dense, Relu, BatchNorm, ScaleShift, Clamp, Div

class Model(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, in_channels, kernel_size, strides=stride, paddings=padding)
        self.min_value = min_value
        self.divisor = divisor

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = Clamp(min=self.min_value)(x)
        x = Div(self.divisor)(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]
