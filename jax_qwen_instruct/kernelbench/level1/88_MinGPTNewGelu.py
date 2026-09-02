import jax
import jax.numpy as jnp
from functools import partial

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @partial(jax.jit, static_argnums=(0,))
    def gelu(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    
    def forward(self, x):
        return self.gelu(x)

batch_size = 8192
dim = 8192

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, dim))]

def get_init_inputs():
    return []
