import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Scale, BatchNorm

class Model:
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.gemm = Dense(out_features)
        self.scale = jnp.ones(scale_shape)
        self.bn = BatchNorm(eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
