import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, BatchNorm, Scale, Softmax

class Model:
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        self.gemm = Dense(out_features)
        self.bn = BatchNorm(axis=-1, epsilon=bn_eps, momentum=bn_momentum)
        self.scale = jax.nn.ones(scale_shape)
        self.softmax = Softmax(axis=1)

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_features).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
