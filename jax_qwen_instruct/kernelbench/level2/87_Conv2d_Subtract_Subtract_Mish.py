import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Mish, Dense

class Model(nn.Module):
    """
    Model that performs a convolution, subtracts two values, applies Mish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    @vmap
    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value_1
        x = x - self.subtract_value_2
        x = Mish()(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

get_inputs = lambda: [jnp.random.rand(batch_size, in_channels, height, width)]
get_init_inputs = lambda: [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
