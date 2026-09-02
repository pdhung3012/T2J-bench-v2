import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, Relu, Flatten

def vgg16(num_classes=1000):
    net = [
        Conv(3, 64, 3, padding=1), Relu,
        Conv(64, 64, 3, padding=1), Relu,
        MaxPool(2, 2),
        Conv(64, 128, 3, padding=1), Relu,
        Conv(128, 128, 3, padding=1), Relu,
        MaxPool(2, 2),
        Conv(128, 256, 3, padding=1), Relu,
        Conv(256, 256, 3, padding=1), Relu,
        Conv(256, 256, 3, padding=1), Relu,
        MaxPool(2, 2),
        Conv(256, 512, 3, padding=1), Relu,
        Conv(512, 512, 3, padding=1), Relu,
        Conv(512, 512, 3, padding=1), Relu,
        MaxPool(2, 2),
        Conv(512, 512, 3, padding=1), Relu,
        Conv(512, 512, 3, padding=1), Relu,
        Conv(512, 512, 3, padding=1), Relu,
        MaxPool(2, 2),
        Flatten,
        Dense(512 * 7 * 7, 4096, activation=Relu),
        Dense(4096, 4096, activation=Relu),
        Dense(4096, num_classes)
    ]
    return net

def forward(params, inputs):
    x = inputs
    for layer in vgg16():
        x = layer(x, **params)
    return x

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (10, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
