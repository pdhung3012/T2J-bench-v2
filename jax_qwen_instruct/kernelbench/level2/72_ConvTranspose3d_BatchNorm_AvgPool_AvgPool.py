import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, BatchNorm, AvgPool, Initializer

def init_model(params):
    net = [
        ConvTranspose(params[0], params[1], (params[2],), strides=params[3], padding=params[4]),
        BatchNorm(),
        AvgPool((2,)),
        AvgPool((2,))
    ]
    return net

def model_fn(params, inputs):
    net = init_model(params)
    for layer in net:
        inputs = layer(inputs)
    return inputs

batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
