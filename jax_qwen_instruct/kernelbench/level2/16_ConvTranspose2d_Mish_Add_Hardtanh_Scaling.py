import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, Mish, Hardtanh, Scale

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = Mish()(x)  # Mish activation
        x = x + self.add_value
        x = Hardtanh(lower=-1., upper=1.)(x)  # Hardtanh activation
        x = self.scale(x)  # Scaling
        return x

batch_size = 128
in_channels = 64  
out_channels = 64  
height = width = 128  
kernel_size = 3
stride = 2  
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
