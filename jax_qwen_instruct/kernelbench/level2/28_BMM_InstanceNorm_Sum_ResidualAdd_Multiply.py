import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, InstanceNorm, Sum, Add, Mul

def model(in_features, out_features, eps=1e-5, momentum=0.1):
    net = [
        Dense(out_features),
        InstanceNorm(out_features, eps=eps, momentum=momentum),
        Sum(),
        Add(),
        Mul()
    ]
    return net

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [jnp.random.rand(batch_size, in_features), jnp.random.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
