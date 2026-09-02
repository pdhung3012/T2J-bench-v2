import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, BatchNorm, Gelu, Relu

class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.batch_norm = BatchNorm()

    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, in_features).
        Returns:
            jax.numpy.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = jnp.gelu(x)
        x = jnp.relu(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [jnp.random.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
