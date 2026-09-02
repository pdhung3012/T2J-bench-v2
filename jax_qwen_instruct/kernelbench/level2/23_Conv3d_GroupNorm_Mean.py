import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, GroupNorm, Mean

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size, kernel_size), padding='SAME')
        self.group_norm = GroupNorm(num_groups)

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_channels, D, H, W).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = jnp.mean(x, axis=(1, 2, 3)) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
