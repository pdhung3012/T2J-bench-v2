import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(jnp.ones(bias_shape))

    def forward(self, x, add_input):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, D, H, W).
            add_input (jnp.ndarray): Input array to be added after transposed convolution, of shape (batch_size, out_channels, D, H, W).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_channels, D, H, W) after HardSwish activation.
        """
        x = self.conv_transpose(x)
        x = x + add_input
        x = x * jax.nn.functional.hardswish(x)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

get_inputs = lambda: [jax.random.normal(key, (batch_size, in_channels, D, H, W)), jax.random.normal(key, (batch_size, out_channels, D*stride, H*stride, W*stride))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
