import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, MaxPool, Conv, Flatten, Initializer, XavierNormal, Sequential

def vgg19(num_classes=1000):
    net = Sequential([
        Conv(3, 64, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(64, 64, (3, 3), padding=(1, 1)),
        Relu(),
        MaxPool((2, 2), strides=(2, 2)),

        Conv(64, 128, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(128, 128, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(128, 128, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(128, 128, (3, 3), padding=(1, 1)),
        Relu(),
        MaxPool((2, 2), strides=(2, 2)),

        Conv(128, 256, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(256, 256, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(256, 256, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(256, 256, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(256, 256, (3, 3), padding=(1, 1)),
        Relu(),
        MaxPool((2, 2), strides=(2, 2)),

        Conv(256, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        MaxPool((2, 2), strides=(2, 2)),

        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        Conv(512, 512, (3, 3), padding=(1, 1)),
        Relu(),
        MaxPool((2, 2), strides=(2, 2)),

        Flatten(),

        Dense(4096),
        Relu(),
        Dropout(rate=0.0),

        Dense(4096),
        Relu(),
        Dropout(rate=0.0),

        Dense(num_classes)
    ])
    return net

def get_inputs():
    return [jnp.random.rand(*batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
