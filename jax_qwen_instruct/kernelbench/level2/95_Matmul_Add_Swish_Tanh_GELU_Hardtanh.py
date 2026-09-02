import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(jnp.zeros(add_value_shape))

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = jnp.sigmoid(x) * x  # Swish
        x = jnp.tanh(x)
        x = jax.nn.gelu(x)  # GELU
        x = jnp.clip(x, -1, 1)  # Hardtanh
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [random.normal(random.PRNGKey(0), (batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
