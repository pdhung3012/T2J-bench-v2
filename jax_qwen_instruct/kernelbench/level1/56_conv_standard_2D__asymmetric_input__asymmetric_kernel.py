import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, BatchNorm, ScaleShift

class Model:
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.model = [
            Conv(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, use_bias=self.bias),
            Relu(),
            BatchNorm(),
            ScaleShift()
        ]
        
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 2D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        for layer in self.model:
            x = layer(x)
        return x

# Test code
batch_size = 8
in_channels = 64
out_channels = 128
kernel_size = (5, 7)
height = 512
width = 256

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
