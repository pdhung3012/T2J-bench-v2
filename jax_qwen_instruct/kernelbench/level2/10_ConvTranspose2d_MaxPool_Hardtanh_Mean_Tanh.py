import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, MaxPool, Hardtanh, Mean, Tanh, Init, Dense

in_channels = 64  
out_channels = 64  
height = width = 256  
kernel_size = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

model = lambda: [
    ConvTranspose(in_channels, out_channels, kernel_size, strides=stride, paddings=padding),
    MaxPool(maxpool_kernel_size, maxpool_stride),
    Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max),
    Mean((2, 3), keepdims=True),
    Tanh()
]

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, height, width)]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
