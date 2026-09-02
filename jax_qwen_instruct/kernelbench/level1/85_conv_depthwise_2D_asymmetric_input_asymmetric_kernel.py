import jax
import jax.numpy as jnp
from jax.experimental import sparse

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = sparse.Conv2D(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, use_bias=bias)
        
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv2d(x)

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (32, 128, 128, 256))
    return [x]

def get_init_inputs():
    return [128, 128, 3, 7, 1, 1, 0, 0, 1, 1, 128, 1]
