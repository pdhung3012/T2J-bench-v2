import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_features: int
    out_features: int
    num_groups: int
    multiply_weight_shape: tuple

    @nn.compact
    def __init__(self, *, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Dense(out_features, in_features=in_features)
        self.group_norm = nn.GroupNorm(num_groups, channels=out_features)
        self.multiply_weight = nn.Parameter(shape=multiply_weight_shape) 

    def __call__(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = x * jnp.sigmoid(x)
        x = x * self.multiply_weight
        x = x * jnp.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, num_groups, multiply_weight_shape]
