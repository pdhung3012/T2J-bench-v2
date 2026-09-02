import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_features: int
    out_features: int
    multiplier: float
    negative_slope: float

    def setup(self):
        self.gemm = nn.Dense(self.out_features)
        self.leaky_relu = nn.relu

    def __call__(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

batch_size = 1024
in_features = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

get_inputs = lambda: jnp.random.rand(batch_size, in_features).astype(jnp.float32)

get_init_inputs = lambda: (in_features, out_features, multiplier, negative_slope)
