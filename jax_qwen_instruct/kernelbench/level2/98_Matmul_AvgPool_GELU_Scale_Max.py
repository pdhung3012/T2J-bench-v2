import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, AvgPool1D, Gelu, Scale, Max

class Model(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = Dense(out_features)
        self.avg_pool = AvgPool1D(pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_features).

        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.avg_pool(x[:, None]).squeeze(axis=-2)
        x = Gelu()(x)
        x = x * self.scale_factor
        x, _ = Max(axis=1, keepdims=False)(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [jnp.random.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
