import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, MaxPool1D, Conv, Relu, Flatten

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, features, sequence_length))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(Model, self).__init__()
        self.maxpool = MaxPool1D(
            kernel_size=kernel_size,
            stride=stride if stride is not None else kernel_size,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices
        )

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.maxpool(x)

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3

return_indices = False

get_inputs = lambda: [jax.random.normal(jax.random.PRNGKey(0), (batch_size, features, sequence_length))]
get_init_inputs = lambda: [kernel_size, stride, padding, dilation, return_indices]
