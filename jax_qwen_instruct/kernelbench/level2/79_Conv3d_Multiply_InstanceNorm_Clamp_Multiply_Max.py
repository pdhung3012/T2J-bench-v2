import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental.stax import Conv, InstanceNorm, Dense, Relu, Flatten, MaxPool, BatchNorm, ScaleShift, Elementwise, Clip

class Model:
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.multiplier_shape = multiplier_shape
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        net = [
            Conv(self.in_channels, self.out_channels, self.kernel_size),
            Dense(self.out_channels),
            Elementwise(lambda x: x * jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(random.random(self.multiplier_shape), axis=0), axis=0), axis=0)),
            InstanceNorm(self.out_channels),
            Clip(min=self.clamp_min, max=self.clamp_max),
            Elementwise(lambda x: x * jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(random.random(self.multiplier_shape), axis=0), axis=0), axis=0)),
            MaxPool(2, 2),
            Flatten(),
            Dense(1)
        ]
        self.params, self.state = Model.init_with_output(net, random.PRNGKey(0))

    @staticmethod
    def init_with_output(net, key):
        params, state = Model.apply(params=None, key=key, **net)
        return params, state

    @staticmethod
    def apply(params, key, inputs, **kwargs):
        outputs = inputs
        for layer in net:
            outputs = layer(outputs, **kwargs)
        return outputs

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [jax.random.normal(key=random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
