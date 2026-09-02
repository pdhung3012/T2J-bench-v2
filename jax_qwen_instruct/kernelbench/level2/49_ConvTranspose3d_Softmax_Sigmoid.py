import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    output_padding: int

    def setup(self):
        self.conv_transpose = nn.ConvTranspose3D(self.in_channels, self.out_channels, 
                                                  kernel_size=self.kernel_size, 
                                                  strides=(self.stride,), 
                                                  padding=self.padding, 
                                                  output_padding=self.output_padding)
        self.softmax = nn.Softmax(axis=1)
        self.sigmoid = nn.Sigmoid()

    @nn.compact
    def __call__(self, x):
        x = self.conv_transpose(x)
        x = self.softmax(x)
        x = self.sigmoid(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

get_inputs = jax.jit(vmap(lambda x: x))

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
