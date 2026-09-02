import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import huber

class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, predictions, targets):
        return huber(predictions - targets, delta=1.0)

batch_size = 32768
input_shape = (32768,)
dim = 1

get_inputs = jax.jit(get_inputs)
get_init_inputs = jax.jit(get_init_inputs)

def get_inputs():
    scale = jnp.random.rand()
    return [jnp.random.rand(batch_size, *input_shape) * scale, jnp.random.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []
