import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

class Model(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = jnp.max(x, axis=self.max_dim, keepdims=True).squeeze(axis=-2)
        x = x - x.mean(axis=1, keepdims=True)
        x = jax.nn.gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]

get_init_inputs = lambda: [in_features, out_features, max_dim]
