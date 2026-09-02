import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose2D, Dense, Relu, Flatten, BatchNorm, MaxPool, ScaleShift

class Model:
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.layers = [
            ConvTranspose2D(self.in_channels, self.out_channels, self.kernel_size, strides=self.stride, paddings=self.padding, outputs=self.output_padding, dilations=self.dilation, groups=self.groups, has_bias=self.bias),
            BatchNorm(),
            Relu(),
            Flatten()
        ]
        
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the transposed 2D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels * height_out * width_out).
        """
        for layer in self.layers:
            x = layer(x)
        return x

# Test code
batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = (3, 5)
height_in = 128
width_in = 256

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
