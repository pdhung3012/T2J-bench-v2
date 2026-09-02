import jax
import jax.numpy as jnp
from jax import vmap
import flax.linen as nn

class Model(nn.Module):
    in_features: int
    out_features: int
    bias_shape: tuple
    num_groups: int

    def setup(self):
        self.gemm = nn.Dense(self.out_features)
        self.bias = nn.Parameter(jnp.zeros(self.bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(self.num_groups, self.out_features)

    @nn.compact
    def __call__(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, bias_shape, num_groups]
