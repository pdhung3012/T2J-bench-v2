import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

    @vmap
    def forward(self, x):
        """
        Args:
            x (jax.numpy.ndarray): Input array of shape (batch_size, in_features).

        Returns:
            jax.numpy.ndarray: Output array of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.max_pool(x[:, None]).squeeze()
        x = jnp.sum(x, axis=1)
        x = x * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

get_inputs = lambda: [jnp.random.rand(batch_size, in_features).astype(jnp.float32)]

get_init_inputs = lambda: [in_features, out_features, kernel_size, scale_factor]
