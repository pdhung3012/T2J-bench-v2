import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose3d, LayerNorm, AvgPool3d, GELU

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose3d(in_channels, out_channels, kernel_size, strides=stride, paddings=padding, outputs=output_padding)
        self.sum_weight = jnp.array(sum_weight)
        self.norm = LayerNorm(norm_shape)
        self.avg_pool = AvgPool3d(pool_kernel_size)
        self.gelu = GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
