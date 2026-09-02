import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, ConvTranspose1D, Relu, BatchNorm, Flatten

def model(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
    net = [
        ConvTranspose1D(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
        Relu(),
        BatchNorm(),
    ]
    if bias:
        net.append(Dense(1))
    return net

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 32, 131072))
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]
