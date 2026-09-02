import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Mish

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = Dense(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = Mish()(x)
        x = Mish()(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

get_inputs = jax.jit(vmap(lambda: jnp.random.rand(batch_size, in_features)))

get_init_inputs = (in_features, out_features)
