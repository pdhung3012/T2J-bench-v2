import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(jnp.random.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, input_size).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, hidden_size).
        """
        x = jnp.matmul(x, self.weight.T)  # Gemm
        x = x / 2  # Divide
        x = jnp.sum(x, axis=1, keepdims=True) # Sum
        x = x * self.scaling_factor  # Scaling
        return x

batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size))]

get_init_inputs = lambda: [input_size, hidden_size, scaling_factor]
