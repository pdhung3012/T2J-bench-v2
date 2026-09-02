import jax
import jax.numpy as jnp

class Model:
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (jax.numpy.ndarray): Input 4D tensor of shape (b, i, j, l)
        B (jax.numpy.ndarray): Input matrix of shape (l, k)

    Returns:
        jax.numpy.ndarray: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        pass

    @staticmethod
    def forward(A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (jax.numpy.ndarray): Input 4D tensor of shape (b, i, j, l)
            B (jax.numpy.ndarray): Input matrix of shape (l, k)

        Returns:
            jax.numpy.ndarray: Output 4D tensor of shape (b, i, j, k)
        """
        return jnp.einsum("bijl,lk->bijk", A, B)

# Test code
b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(b, i, j, l))
    B = jax.random.normal(key=jax.random.PRNGKey(1), shape=(l, k))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
