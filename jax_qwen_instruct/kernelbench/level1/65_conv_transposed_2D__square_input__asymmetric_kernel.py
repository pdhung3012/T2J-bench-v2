import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose2D, Dense, Relu, Flatten, BatchNorm, MaxPool2D, ScaleShift

class Model:
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.model = [
            ConvTranspose2D(in_channels, out_channels, kernel_size, strides=self.stride, paddings=self.padding, outputs=self.output_padding, groups=self.groups, has_bias=self.bias),
            BatchNorm(),
            Relu(),
            ConvTranspose2D(out_channels, out_channels, kernel_size=(1, 1), strides=(1, 1), paddings=(0, 0), outputs=(0, 0), groups=1, has_bias=False),
            BatchNorm(),
            Relu()
        ]

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the transposed 2D convolution.

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
out_channels = 64
kernel_size = (3, 7)  # larger asymmetric kernel
width = 512
height = 512

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
