import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, BatchNorm, Scale, GlobalAvgPool3d

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, in_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = BatchNorm(out_channels, epsilon=eps, momentum=momentum)
        self.global_avg_pool = GlobalAvgPool3d()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
