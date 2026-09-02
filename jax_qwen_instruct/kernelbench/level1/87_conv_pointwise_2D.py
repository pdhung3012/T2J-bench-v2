import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, Relu, MaxPool, Flatten, BatchNorm

class Model(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = Conv(1, out_channels, padding=0, bias=bias)
        
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the pointwise 2D convolution.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv1d(x)

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
