import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, InstanceNorm, Relu, Flatten

def model(num_features: int) -> jax.nn.Module:
    """
    Simple model that performs Instance Normalization.
    """
    net = [
        Dense(num_features),
        InstanceNorm(),
        Relu(),
        Flatten()
    ]
    return jax.nn.Module(vmap(jax.jit(jax.vjp)(lambda x: x, net)))

batch_size = 112  # heavier workload
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
