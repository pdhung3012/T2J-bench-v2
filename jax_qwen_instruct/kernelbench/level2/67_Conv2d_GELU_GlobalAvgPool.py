import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Gelu, AdaptiveAvgPool2d, Flatten

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = Gelu()(x)
        x = AdaptiveAvgPool2d(1)(x)
        x = jnp.squeeze(x, axis=(-1, -2))
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
