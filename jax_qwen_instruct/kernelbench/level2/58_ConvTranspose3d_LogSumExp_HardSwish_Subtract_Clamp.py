import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(jnp.ones(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = jnp.logsumexp(x, axis=1, keepdims=True)
        x = x * jnp.sigmoid(x + 3) / 6
        x = x - self.bias
        x = jnp.clip(x, a_min=-1, a_max=1)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
