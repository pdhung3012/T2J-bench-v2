import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Relu, LeakyRelu, GELU, Sigmoid, Param, Dense

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size, kernel_size), padding='SAME')
        self.bias = Param(jnp.zeros(bias_shape))

    def init_weights_and_state(self, rng, input_shape):
        key, *rest_keys = jax.random.split(rng, 1 + len(input_shape))
        return self.init(key, input_shape), rest_keys

    def apply(self, params, state, inputs):
        x = self.conv(inputs)
        x = Relu()(x)
        x = LeakyRelu(0.01)(x)
        x = GELU()(x)
        x = Sigmoid()(x)
        x = x + params['bias']
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

get_inputs = vmap(lambda x: x)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
