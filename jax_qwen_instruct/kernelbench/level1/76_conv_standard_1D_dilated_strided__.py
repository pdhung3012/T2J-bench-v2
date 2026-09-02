import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv1D, Dense, Relu, Flatten, MaxPool1D, BatchNorm, ScaleShift

class Model:
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        net = [
            Conv1D(self.in_channels, self.out_channels, self.kernel_size, padding='SAME', strides=self.stride, dilation_rate=self.dilation, use_bias=self.bias),
            BatchNorm(),
            Relu(),
            # Flatten()  # Not needed if you want to keep the last dimension for other operations
        ]
        self.net = jax.experimental.stax.serial(*net)
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the 1D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.net(x)

# Test code
batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
# longer signal
length = 524280
stride = 3
dilation = 4

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
