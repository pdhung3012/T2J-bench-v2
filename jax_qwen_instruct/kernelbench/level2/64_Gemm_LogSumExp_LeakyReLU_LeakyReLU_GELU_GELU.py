import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from jax.experimental.optimizers import clip_grads

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    @vmap
    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # LogSumExp
        x = logsumexp(x, axis=1, keepdims=True)
        # LeakyReLU
        x = jax.nn.leaky_relu(x, alpha=0.01)
        # LeakyReLU
        x = jax.nn.leaky_relu(x, alpha=0.01)
        # GELU
        x = jax.nn.gelu(x)
        # GELU
        x = jax.nn.gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

get_inputs = lambda: [jnp.random.rand(batch_size, in_features)]
get_init_inputs = lambda: [in_features, out_features]
