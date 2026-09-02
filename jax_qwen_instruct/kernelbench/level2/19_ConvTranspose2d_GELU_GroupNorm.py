import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, Gelu, GroupNorm, Dense

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, kernel_size, stride=stride)
        self.group_norm = GroupNorm(num_groups=num_groups, channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = Gelu()(x)
        x = self.group_norm(x)
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
