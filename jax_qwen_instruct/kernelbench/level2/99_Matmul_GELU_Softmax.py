import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    @vmap
    def forward(self, x):
        x = self.linear(x)
        x = jax.nn.gelu(x)
        x = jax.nn.softmax(x, axis=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [jnp.random.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
