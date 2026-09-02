import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental.stax import Dense, BatchNorm, BiasAdd, Div, Swish

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(Model, self).__init__()
        self.matmul = Dense(out_features)
        self.bn = BatchNorm(eps=bn_eps, momentum=bn_momentum)
        self.bias = jnp.ones(bias_shape)
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * Swish()(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [random.normal(random.PRNGKey(0), (batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
