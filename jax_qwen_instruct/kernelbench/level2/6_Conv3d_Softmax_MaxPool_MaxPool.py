import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, Softmax, Init

class Model:
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        self.params = Init(
            Conv(in_channels, out_channels, kernel_size),
            MaxPool(pool_kernel_size),
            MaxPool(pool_kernel_size),
            Dense(out_channels, activation=Softmax())
        )
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, x, params):
        return params.apply(params, x)

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [jnp.random.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
