import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (jnp.ndarray): Input array of shape (batch_size, in_features).
        Returns:
            jnp.ndarray: Output array of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = jnp.sum(x, axis=1, keepdims=True) # (batch_size, 1)
        x = jnp.max(x, axis=1, keepdims=True)[0] # (batch_size, 1)
        x = jnp.mean(x, axis=1, keepdims=True) # (batch_size, 1)
        x = jnp.log(jnp.sum(jnp.exp(x), axis=1, keepdims=True)) # (batch_size, 1)
        x = jnp.log(jnp.sum(jnp.exp(x), axis=1, keepdims=True)) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

get_inputs = jax.jit(vmap(get_inputs))

get_init_inputs = jax.jit(vmap(get_init_inputs))
