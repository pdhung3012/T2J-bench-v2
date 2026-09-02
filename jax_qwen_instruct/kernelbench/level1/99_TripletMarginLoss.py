import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = jax.nn.triangular_margin_loss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    scale = jax.random.uniform(jax.random.PRNGKey(0), ())
    return [jax.random.uniform(jax.random.PRNGKey(1), (batch_size, *input_shape))*scale,
            jax.random.uniform(jax.random.PRNGKey(2), (batch_size, *input_shape)),
            jax.random.uniform(jax.random.PRNGKey(3), (batch_size, *input_shape))]

def get_init_inputs():
    return [1.0]  # Default margin
