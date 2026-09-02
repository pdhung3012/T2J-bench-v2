import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose3D, Dense, Relu, Flatten, BatchNorm, MaxPool3D, ScaleShift

class Model:
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

        net = [
            ConvTranspose3D(self.in_channels, self.out_channels, self.kernel_size, strides=self.stride, paddings=self.padding, outputs=self.output_padding, groups=self.groups, has_bias=self.bias),
            BatchNorm(),
            Relu(),
            ConvTranspose3D(self.out_channels, self.out_channels, self.kernel_size, strides=self.stride, paddings=self.padding, outputs=self.output_padding, groups=self.groups, has_bias=self.bias),
            BatchNorm(),
            Relu()
        ]
        self.net = jax.experimental.stax.serial(*net)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the transposed 3D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.net(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, width, height))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization
