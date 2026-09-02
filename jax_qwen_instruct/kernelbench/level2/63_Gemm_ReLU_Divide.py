import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        x = self.linear(x)
        x = jnp.maximum(0, x)  # ReLU equivalent in JAX
        x = x / self.divisor
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

get_init_inputs = lambda: [in_features, out_features, divisor]
