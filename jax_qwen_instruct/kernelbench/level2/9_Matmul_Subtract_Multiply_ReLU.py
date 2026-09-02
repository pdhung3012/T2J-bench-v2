import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = jnp.maximum(0, x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, subtract_value, multiply_value]
