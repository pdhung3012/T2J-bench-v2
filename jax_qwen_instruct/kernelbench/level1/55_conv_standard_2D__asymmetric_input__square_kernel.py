import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, BatchNorm, ScaleShift

class Model:
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.model = [
            Conv(self.in_channels, self.out_channels, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias),
            Relu(),
            Flatten()
        ]
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 2D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels * height_out * width_out).
        """
        for layer in self.model:
            x = layer(x)
        return x

# Test code
batch_size = 8
# smaller spatial dims
height = 512
width = 1024
in_channels = 64  # increased channels
out_channels = 128
kernel_size = 3
# asymmetric input: make width considerably larger than height

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
