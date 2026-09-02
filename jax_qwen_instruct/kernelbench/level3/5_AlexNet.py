import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, Relu, Flatten

model = [
    Conv(96, (11, 11), strides=(4, 4), padding=(2, 2)),
    Relu,
    MaxPool((3, 3), strides=(2, 2)),
    Conv(256, (5, 5), padding=(2, 2)),
    Relu,
    MaxPool((3, 3), strides=(2, 2)),
    Conv(384, (3, 3), padding=(1, 1)),
    Relu,
    Conv(384, (3, 3), padding=(1, 1)),
    Relu,
    Conv(256, (3, 3), padding=(1, 1)),
    Relu,
    MaxPool((3, 3), strides=(2, 2)),
    Flatten,
    Dense(4096),
    Relu,
    Dense(4096),
    Relu,
    Dense(4096),
    Dense(num_classes)
]

def model_fn(features, params, batch_size=1024, num_classes=1000):
    for layer in model[:-1]:
        features = layer(features)
    logits = model[-1](features)
    return logits

def get_inputs():
    return [jnp.random.rand(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
