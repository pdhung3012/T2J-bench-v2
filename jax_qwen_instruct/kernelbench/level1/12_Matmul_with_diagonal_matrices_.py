import jax
import jax.numpy as jnp

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (jnp.ndarray): A 1D array representing the diagonal of the diagonal matrix. Shape: (N,).
            B (jnp.ndarray): A 2D array representing the second matrix. Shape: (N, M).

        Returns:
            jnp.ndarray: The result of the matrix multiplication. Shape: (N, M).
        """
        # Logically equivalent to jnp.diag(A) @ B 
        # more efficient as no need to materialize a full N×N matrix
        return jnp.expand_dims(A, axis=1) * B

M = 4096
N = 4096

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(N,))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(N, M))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
