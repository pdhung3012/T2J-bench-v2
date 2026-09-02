import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose2D, Dense, Relu, Flatten, BatchNorm, MaxPool2D, ScaleShift

class Model:
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        self.model = [
            ConvTranspose2D(in_channels, out_channels, (kernel_size, kernel_size), strides=(stride, stride), padding=((padding, padding), (padding, padding)), output_padding=(output_padding, output_padding), groups=groups, bias=bias),
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
        for layer in self.model:
            x = layer(x)
        return x

# Test code
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = 3
# large asymmetric input
height_in = 512
width_in = 1024

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
