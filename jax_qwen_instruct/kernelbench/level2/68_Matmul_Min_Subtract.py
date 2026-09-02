import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_features: int
    out_features: int
    constant: float

    def setup(self):
        self.linear = nn.Dense(self.out_features, use_bias=False)
        self.constant = self.param('constant', jnp.zeros, ())

    @nn.compact
    def __call__(self, x):
        x = self.linear(x)
        x = jnp.minimum(x, self.constant)
        x = x - self.constant
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

get_init_inputs = lambda: [in_features, out_features, constant]
