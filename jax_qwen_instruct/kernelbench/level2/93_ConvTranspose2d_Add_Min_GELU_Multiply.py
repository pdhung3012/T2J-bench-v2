import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, Add, Min, Gelu, Scale

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.add_value
        x = jnp.minimum(x, 0.0)
        x = Gelu()(x)
        x = x * self.multiply_value
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
