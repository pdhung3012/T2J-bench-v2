import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, AvgPool, Dense, Relu, Sigmoid

def model(in_channels, out_channels, kernel_size, pool_kernel_size):
    net = [
        Conv(in_channels, out_channels, (kernel_size, kernel_size)),
        AvgPool((pool_kernel_size, pool_kernel_size)),
        Relu,
        Sigmoid,
        lambda x: jnp.sum(x, axis=(1,2,3))
    ]
    return net

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
