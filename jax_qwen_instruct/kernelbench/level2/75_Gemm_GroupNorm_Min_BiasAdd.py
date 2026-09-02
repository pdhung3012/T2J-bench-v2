import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, GroupNorm, Min, Add, Param

class Model(nn.Module):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.group_norm = GroupNorm(num_groups, out_features)
        self.bias = Param(jnp.zeros(bias_shape))

    def init_params(self, batch_size, in_features):
        return {
            'gemm': self.gemm.init(jax.random.PRNGKey(0), jnp.ones((batch_size, in_features))),
            'group_norm': self.group_norm.init(jax.random.PRNGKey(1), self.gemm.W),
            'bias': self.bias.init(jax.random.PRNGKey(2))
        }

    def apply(self, params, batch_size, in_features, x):
        x = self.gemm.apply(params['gemm'], x)
        x = self.group_norm.apply(params['group_norm'], x)
        x = jnp.minimum(x, axis=1, keepdims=True)[0]
        x = x + params['bias']
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(3), shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
