import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, HardSwish, GroupNorm, MeanPool, Init

class Model(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size, kernel_size), padding='SAME', bias=bias)
        self.group_norm = GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = HardSwish()(x)                           # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = MeanPool((1, 2, 3))(x)                   # Mean over spatial dims → (B, C)
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
