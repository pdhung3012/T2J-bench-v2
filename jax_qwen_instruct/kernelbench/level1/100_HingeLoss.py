import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    @jax.jit
    def forward(self, predictions, targets):
        return jnp.mean(jnp.clip(1 - predictions * targets, min=0))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs(seed):
    key = random.PRNGKey(seed)
    inputs = random.normal(key, shape=(batch_size, *input_shape))
    targets = random.randint(key, shape=(batch_size,), minval=0, maxval=2) * 2 - 1
    return inputs, targets

def get_init_inputs():
    return []
