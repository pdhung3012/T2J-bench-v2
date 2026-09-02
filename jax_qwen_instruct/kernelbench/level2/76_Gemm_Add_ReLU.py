import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(jnp.zeros(bias_shape))

    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array with shape (batch_size, in_features).
        Returns:
            jax.numpy.ndarray: Output array with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = jnp.maximum(0, x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, bias_shape]
