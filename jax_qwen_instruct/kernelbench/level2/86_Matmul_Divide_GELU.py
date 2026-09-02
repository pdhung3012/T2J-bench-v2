import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Gelu

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = Dense(output_size)
        self.divisor = divisor

    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, input_size).
        Returns:
            jax.numpy.ndarray: Output array of shape (batch_size, output_size).
        """
        x = self.linear(x)
        x = x / self.divisor
        x = Gelu()(x)
        return x

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, output_size, divisor]
