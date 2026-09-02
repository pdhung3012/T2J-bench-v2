import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Param, Relu, Add, GlobalAvgPool, LogSumExp, Tanh

class Model:
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, with_bias=bias)
        self.subtract = Param(jnp.zeros(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = self.gemm(x)

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = jnp.mean(x, axis=1, keepdims=True)

        # LogSumExp
        x = LogSumExp()(x)

        # GELU (approximated using Tanh)
        x = Relu()(x) * (1 + jnp.tanh((2 * x + 1) / jnp.sqrt(2))) / 2

        # ResidualAdd
        x = x + original_x

        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
