import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Flatten, BatchNorm, Relu, LogSoftmax

class Model:
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        net = [
            Conv(in_channels, out_channels, kernel_size, strides=self.stride, paddings=self.padding, dilations=self.dilation, groups=self.groups, has_bias=self.bias),
            Relu(),
            # Add more layers if needed
        ]
        self.model = jax.experimental.stax.serial(*net)
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 3D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.model(x)

# Test code
batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 128
width = 128

def get_inputs():
    x = jnp.random.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
