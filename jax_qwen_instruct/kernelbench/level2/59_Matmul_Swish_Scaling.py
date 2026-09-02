import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = x * jax.nn.sigmoid(x)  # Swish activation
        x = x * self.scaling_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

get_init_inputs = lambda: [in_features, out_features, scaling_factor]
