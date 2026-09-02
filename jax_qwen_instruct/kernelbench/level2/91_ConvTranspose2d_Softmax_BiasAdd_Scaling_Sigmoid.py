import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(jnp.zeros(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = jnp.softmax(x, axis=1)
        x = x + self.bias
        x = x * self.scaling_factor
        x = jnp.clip(jnp.sigmoid(x), a_min=0., a_max=1.)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs(seed):
    key = random.PRNGKey(seed)
    x = random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
