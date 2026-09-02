import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, GroupNorm, Min, Clamp, Dropout

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size, kernel_size), padding='SAME')
        self.norm = GroupNorm(groups)
        self.dropout = Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = jnp.minimum(x, jnp.array(min_value))
        x = Clamp(min=min_value, max=max_value)(x)
        x = self.dropout(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
