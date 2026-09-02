import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_features).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x * jnp.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = jnp.clip(x, a_min=-1.0, a_max=1.0)  # Clamp between -1 and 1
        x = jnp.tanh(x)  # Tanh activation
        x = jnp.clip(x, a_min=-1.0, a_max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

get_inputs = lambda: [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, in_features))]

get_init_inputs = lambda: [in_features, out_features]
