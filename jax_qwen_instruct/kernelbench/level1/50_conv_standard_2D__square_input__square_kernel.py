import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, Relu, Initializer

def model(num_classes=1000):
    init_key = jax.random.PRNGKey(0)
    params = model_init(init_key, (batch_size, 3, 224, 224), num_classes)
    apply_fn = model_apply(params)
    return apply_fn

def model_init(key, input_shape, output_dim):
    net = [
        Conv(96, (11, 11), strides=(4, 4), padding=(2, 2)),
        Relu(),
        MaxPool((3, 3), strides=(2, 2)),
    ]
    return net, key

def model_apply(params, inputs):
    net = params[0]
    for layer in net:
        inputs = layer(inputs)
    return inputs

batch_size = 256
num_classes = 1000

def get_inputs():
    return [jnp.random.rand(batch_size, 3, 224, 224).astype(jnp.float32)]

def get_init_inputs():
    return [num_classes]
