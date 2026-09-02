import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose2D, Dense, Relu, Flatten, BatchNorm, MaxPool, Concatenate

class Model:
    """
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        net = [
            ConvTranspose2D(in_channels, out_channels, kernel_size, strides=self.stride, paddings=self.padding, with_bias=self.bias),
            BatchNorm(),
            Relu(),
            Flatten()
        ]
        self.net = jax.experimental.stax.serial(*net)
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 2D transposed convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels * height * width).
        """
        return self.net(x)

# Test code
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
