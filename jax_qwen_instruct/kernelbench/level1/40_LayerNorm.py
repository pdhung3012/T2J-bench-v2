import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    @partial(jit, static_argnums=(0,))
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (jnp.ndarray): Input tensor of shape (*, normalized_shape).

        Returns:
            jnp.ndarray: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

get_inputs = jit(vmap(get_inputs))

get_init_inputs = jit(vmap(lambda s: (s[0], s[1], s[2])))
