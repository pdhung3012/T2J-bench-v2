import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, AdaptiveAvgPool, Dense, Add, Sum

def model(in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
    net = [
        Conv(in_channels, out_channels, kernel_size),
        lambda x: x / divisor,
        MaxPool(pool_size),
        AdaptiveAvgPool((1, 1, 1)),
        Dense(bias_shape),
        Add(),
        Sum(sum_dim)
    ]
    return net

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
