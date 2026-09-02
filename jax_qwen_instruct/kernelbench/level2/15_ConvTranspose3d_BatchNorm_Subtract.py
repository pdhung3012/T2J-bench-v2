import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, BatchNorm, Dense, Relu, Flatten, Softmax, MaxPool, Input, Add

class Model:
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.params = {}

    def init_params(self, input_shape):
        net = [
            ConvTranspose(self.in_channels, self.out_channels, self.kernel_size, strides=self.stride, padding=self.padding),
            BatchNorm(self.out_channels),
            lambda x: x - jnp.mean(x, axis=(2, 3, 4), keepdims=True)  # Subtract mean along spatial dimensions
        ]
        net = Input(input_shape) >> net >> Dense(10) >> Relu() >> Flatten() >> Dense(10) >> Relu() >> Dense(1) >> Softmax()
        self.params = net.init(jax.random.PRNGKey(0), jnp.ones(input_shape))

    def forward(self, x):
        for layer in self.params['params']:
            x = layer(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(1), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
