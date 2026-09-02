import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, MaxPool, AdaptiveAvgPool3d, Scale, Clamp

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, in_channels, kernel_size, strides=stride, paddings=padding)
        self.scale = Scale(scale)
        self.maxpool = MaxPool(maxpool_kernel_size)
        self.global_avg_pool = AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.scale(x)
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = Clamp(min=self.clamp_min, max=self.clamp_max)(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
