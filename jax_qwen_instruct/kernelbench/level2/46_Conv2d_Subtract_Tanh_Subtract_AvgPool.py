import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, Tanh, AvgPool, Initializer

def model(in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
    net = [
        Conv(in_channels, out_channels, (kernel_size, kernel_size), padding='SAME'),
        lambda x: x - subtract1_value,
        Tanh(),
        lambda x: x - subtract2_value,
        AvgPool((kernel_size_pool, kernel_size_pool)),
    ]
    return net

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
