import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, GroupNorm, Relu, Flatten

def model(num_features: int, num_groups: int) -> jax.nn.Module:
    """
    Simple model that performs Group Normalization.
    """
    net = [
        Dense(num_features),
        GroupNorm(num_groups=num_groups, num_channels=num_features),
        Relu(),
        Flatten()
    ]
    return jax.nn.ModuleList(net)

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features
