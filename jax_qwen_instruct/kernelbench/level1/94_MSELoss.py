import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    @jax.jit
    def forward(self, predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs(seed):
    key = random.PRNGKey(seed)
    scale = random.normal(key, ())
    return [random.normal(key, (batch_size, *input_shape)) * scale, random.normal(key, (batch_size, *input_shape))]

def get_init_inputs():
    return []
