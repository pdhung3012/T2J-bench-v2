import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose2D, Dense, Relu, Flatten, BatchNorm, MaxPool, ScaleShift

class Model:
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

        net = [
            ConvTranspose2D(in_channels, out_channels, (kernel_size, kernel_size), strides=(stride, stride), paddings=((padding, padding), (padding, padding)), dilations=((dilation, dilation), (dilation, dilation)), with_bias=bias),
            BatchNorm(),
            Relu(),
            Flatten()
        ]
        self.net = jax.experimental.stax.serial(*net)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 2D transposed convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels * height_out * width_out).
        """
        return self.net(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
