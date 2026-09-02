import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, BatchNorm, ScaleShift

def model(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
    net = [
        Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
        Relu(),
        BatchNorm(),
        ScaleShift()
    ]
    return net

def forward_fn(params, state, inputs):
    for layer in model(*get_init_inputs()):
        inputs = layer(inputs, **params)
    return inputs

batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512

inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
params, state = {}, {}

print(forward_fn(params, state, inputs))
