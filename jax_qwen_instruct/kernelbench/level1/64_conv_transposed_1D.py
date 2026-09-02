import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, ConvTranspose1D, Relu

def model(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
    net = [
        ConvTranspose1D(out_channels, kernel_size, strides=stride, paddings=padding, outputs=output_padding, groups=groups, has_bias=bias),
        Relu()
    ]
    return Dense(1)

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 128, 65536))
    return [x]

def get_init_inputs():
    return [128, 128, 3]  # Provide in_channels, out_channels, kernel_size for initialization
