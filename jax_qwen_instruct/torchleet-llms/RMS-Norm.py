import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Normalize, Exp, Add, Constant

class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.eps = eps
        self.scale = jnp.ones(dim)  # gamma

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: shape (..., dim)
        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + self.eps)  # RMS
        return (x / norm) * self.scale

x = jnp.randn(3, 5)  # e.g., (batch_size=3, features=5)
rmsnorm = RMSNorm(dim=5)
out = rmsnorm(x)
print(out.shape)  # should be (3, 5)
assert out.shape == (3, 5), "Output shape mismatch"
