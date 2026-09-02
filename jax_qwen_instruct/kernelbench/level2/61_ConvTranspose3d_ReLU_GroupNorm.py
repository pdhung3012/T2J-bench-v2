import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, Relu, GroupNorm

class Model(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, in_channels, kernel_size, padding='SAME', strides=(1, 1, 1), with_bias=bias)
        self.relu = Relu()
        self.group_norm = GroupNorm(groups)

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, D, H, W).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
