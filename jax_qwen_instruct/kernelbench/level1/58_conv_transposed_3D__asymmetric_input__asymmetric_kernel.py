import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    @partial(jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv_transpose3d(x)

def get_inputs():
    batch_size = 16
    in_channels = 32
    out_channels = 16
    kernel_size = (3, 5, 7)  # Asymmetric kernel size
    depth_in = 16
    height_in = 32
    width_in = 64
    x = jnp.random.rand(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth_in = 16
height_in = 32
width_in = 64

x = get_inputs()[0]
model = Model(in_channels, out_channels, kernel_size)
output = model(x)
print(output.shape)
