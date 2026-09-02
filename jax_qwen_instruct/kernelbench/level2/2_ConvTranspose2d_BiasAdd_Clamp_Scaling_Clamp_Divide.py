import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    @nn.compact
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2D(in_channels, out_channels, kernel_size, strides=(stride,), padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(jnp.zeros(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias.vevmap()
        x = jnp.clip(x, a_min=0.0, a_max=1.0)
        x = x * self.scaling_factor
        x = jnp.clip(x, a_min=0.0, a_max=1.0)
        x = x / self.scaling_factor
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
