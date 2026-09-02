import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    @partial(jax.jit, static_argnames=['stride', 'padding', 'dilation', 'groups'])
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv3d(x)

def get_inputs():
    batch_size = 16
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    depth = 64
    width = 64
    height = 64
    x = jnp.random.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
