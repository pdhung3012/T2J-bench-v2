import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv2d(x)

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(16, 16, 1024, 1024))
    return [x]

def get_init_inputs():
    return [16, 128, 3]  # Provide in_channels, out_channels, kernel_size for initialization
