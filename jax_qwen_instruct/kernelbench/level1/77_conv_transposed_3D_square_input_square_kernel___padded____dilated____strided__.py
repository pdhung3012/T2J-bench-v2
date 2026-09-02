import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)

    @partial(jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv_transpose3d(x)

def get_inputs():
    x = jnp.random.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]

batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

get_inputs = vmap(get_inputs, in_axes=(0,))
get_init_inputs = vmap(get_init_inputs, in_axes=(0,))
