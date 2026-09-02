import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: tuple
    sum_tensor_shape: tuple

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.out_channels, kernel_size=self.kernel_size)(x)
        x = jax.nn.leaky_relu(x, alpha=0.2)
        x = x + self.param('sum_tensor', jnp.zeros, self.sum_tensor_shape)
        x = jnp.clip(x, -1.0, 1.0)
        x = jax.nn.gelu(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = (3, 3, 3)
sum_tensor_shape = (out_channels, 1, 1, 1)

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_channels, depth, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, sum_tensor_shape]
