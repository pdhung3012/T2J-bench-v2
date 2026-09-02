import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = jnp.sigmoid(x)
        x = self.linear2(x)
        x = jnp.logsumexp(x, axis=1)  # compute LogSumExp over features per sample
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size))]

get_init_inputs = lambda: [input_size, hidden_size, output_size]
