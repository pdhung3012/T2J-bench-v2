import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, GroupNorm, Scale, MaxPool, BatchNorm, Relu, Dense, Flatten, LogSoftmax

def model_fn(params, inputs):
    net = [
        Conv(8, 64, (3, 3), padding='SAME'),
        GroupNorm(16),
        Scale((64, 1, 1)),
        MaxPool((4, 4)),
        BatchNorm(),
        Relu(),
        Dense(10)
    ]
    net_out = inputs
    for layer in net:
        net_out = layer(net_out)
    return net_out

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (128, 8, 128, 128))]

def get_init_inputs():
    return [8, 64, 3, 16, (64, 1, 1), 4, 0.0, 1.0]
