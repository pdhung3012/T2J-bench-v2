import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, InstanceNorm, Dense

def model(in_channels, out_channels, kernel_size, divide_by):
    net = [
        Conv(in_channels, out_channels, (kernel_size, kernel_size)),
        InstanceNorm(out_channels),
        lambda x: x / divide_by
    ]
    return lambda x: jnp.concatenate([f(x) for f in net], axis=-1)

batch_size = 128
in_channels = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
