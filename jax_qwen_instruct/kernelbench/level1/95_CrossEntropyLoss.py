import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    @vmap
    def forward(self, predictions, targets):
        return jnp.mean(jnp.sum(-targets * jnp.log(predictions + 1e-8), axis=-1))

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, *input_shape)), 
            jax.random.randint(key=jax.random.PRNGKey(1), shape=(batch_size,), minval=0, maxval=num_classes)]

def get_init_inputs():
    return []
