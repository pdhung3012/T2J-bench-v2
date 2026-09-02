import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, Relu, Flatten, MaxPool, BatchNorm, Initializer

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = Conv(out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
