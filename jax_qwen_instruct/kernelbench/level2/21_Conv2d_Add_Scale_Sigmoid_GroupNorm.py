import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, BatchNorm, Sigmoid, Relu

def model(in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
    net = [
        Conv(in_channels, out_channels, (kernel_size, kernel_size)),
        Dense(out_channels),
        BatchNorm(num_groups=num_groups),
        Sigmoid(),
    ]
    return net

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
