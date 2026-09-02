import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    @jit
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of arbitrary shape.

        Returns:
            jnp.ndarray: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = jnp.linalg.norm(x, ord='fro', axis=(-2, -1))
        return x / norm

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

get_inputs = jit(vmap(get_inputs))

def get_init_inputs():
    return []
