import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Relu, MaxPool, Dense, AdapMaxPool, Cat

class FireModule:
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        self.layers = [
            Conv(in_channels, squeeze_channels, kernel_size=1),
            Relu(),
            Conv(squeeze_channels, expand1x1_channels, kernel_size=1),
            Relu(),
            Conv(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            Relu()
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Model:
    def __init__(self, num_classes=1000):
        self.features = [
            Conv(3, 96, kernel_size=7, stride=2),
            Relu(),
            MaxPool(kernel_size=3, stride=2, padding="same"),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            MaxPool(kernel_size=3, stride=2, padding="same"),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            MaxPool(kernel_size=3, stride=2, padding="same"),
            FireModule(512, 64, 256, 256),
        ]
        
        self.classifier = [
            Dense(512, 256),
            Relu(),
            AdapMaxPool(pool_size=1),
            Dense(num_classes)
        ]
    
    def __call__(self, x):
        for layer in self.features:
            x = layer(x)
        x = jnp.concatenate([layer(x) for layer in self.features], axis=-1)
        for layer in self.classifier:
            x = layer(x)
        return x

def get_inputs():
    return [jnp.random.rand(*input_shape)]

def get_init_inputs():
    return [num_classes]

input_shape = (batch_size, input_channels, height, width)
