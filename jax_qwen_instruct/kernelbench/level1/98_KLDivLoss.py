import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import multinomial

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    @partial(jax.jit, static_argnums=(0,))
    def kl_divergence(self, log_probs, probs):
        return jnp.mean(jnp.sum(log_probs - log_probs * probs - jnp.log(jnp.sum(jnp.exp(log_probs), axis=-1)), axis=-1), axis=0)

@partial(jax.jit, static_argnums=(0,))
def get_inputs():
    scale = jnp.random.rand()
    return [vmap(lambda x: x.softmax(-1))(jnp.random.rand(batch_size, *input_shape) * scale),
            vmap(lambda x: x.softmax(-1))(jnp.random.rand(batch_size, *input_shape))]

def get_init_inputs():
    return []

model = Model()
kl_loss = model.kl_divergence(*get_inputs())
