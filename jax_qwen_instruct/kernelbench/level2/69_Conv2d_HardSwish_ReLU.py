import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, HardSwish, Relu

class Model(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, height, width).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = HardSwish()(x)
        x = Relu()(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
