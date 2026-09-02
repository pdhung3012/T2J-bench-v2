import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    @partial(jit, static_argnums=(1, 2, 3, 4, 5, 6))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv_transpose3d(x)

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(4, 32, 32, 64, 128))
    return [x]

def get_init_inputs():
    return [32, 32, 3, 2, 1, 4]
