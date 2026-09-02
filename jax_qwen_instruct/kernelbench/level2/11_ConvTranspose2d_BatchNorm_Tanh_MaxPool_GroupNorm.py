import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, BatchNorm, Tanh, MaxPool, GroupNorm

def model(in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
    net = [
        ConvTranspose(in_channels, out_channels, kernel_size, strides=stride, paddings=padding),
        BatchNorm(),
        Tanh(),
        MaxPool(2, 2),
        GroupNorm(num_groups=num_groups)
    ]
    return jax.experimental.stax.serial(*net)

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [jnp.random.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
