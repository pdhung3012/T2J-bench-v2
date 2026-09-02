import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Hardtanh, Gelu

class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = Gelu()

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scaling_factor
        x = self.hardtanh(x)
        x = self.gelu(x)
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

get_inputs = lambda: jnp.array([jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_features))])

get_init_inputs = lambda: [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
