import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental.stax import Dense, GroupNorm, HardTanh

class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.group_norm = GroupNorm(num_groups, out_features)
        self.hardtanh = HardTanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, in_features).
        Returns:
            jax.numpy.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.group_norm(x)
        x = self.hardtanh(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs(key):
    key, subkey = random.split(key)
    return random.normal(subkey, (batch_size, in_features))

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
