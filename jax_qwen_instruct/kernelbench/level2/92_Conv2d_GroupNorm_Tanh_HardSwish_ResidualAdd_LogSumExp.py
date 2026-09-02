import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Conv, GroupNorm, Tanh, HardSwish, LogSumExp

class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
        self.group_norm = GroupNorm(groups)
        self.tanh = Tanh()
        self.hard_swish = HardSwish()
        self.logsumexp = LogSumExp(keepdims=True)

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = self.logsumexp(x_res)
        return x_logsumexp

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

get_inputs = jax.jit(vmap(get_inputs))
get_init_inputs = jax.jit(vmap(get_init_inputs))
