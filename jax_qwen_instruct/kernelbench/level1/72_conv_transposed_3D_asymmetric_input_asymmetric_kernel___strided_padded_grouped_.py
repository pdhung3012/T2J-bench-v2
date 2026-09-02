import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose3D, Dense, Relu, Flatten, BatchNorm, MaxPool3D, ScaleShift

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        net = [
            ConvTranspose3D(in_channels, out_channels, kernel_size, strides=stride, paddings=padding, outputs=output_padding, groups=groups, has_bias=bias),
            BatchNorm(),
            Relu(),
            ConvTranspose3D(out_channels, out_channels, kernel_size, strides=stride, paddings=padding, outputs=output_padding, groups=groups, has_bias=bias),
            BatchNorm(),
            Relu()
        ]
        self.net = net

    def __call__(self, inputs):
        for layer in self.net:
            inputs = layer(inputs)
        return inputs

# Test code
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 5, 7)
depth = 12
height = 24
width = 48
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4

def get_inputs():
    x = jnp.random.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]
