import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Conv2D, MaxPool2D, BatchNorm, Relu, Dropout

def model(num_layers: int, num_input_features: int, growth_rate: int):
    def init(key):
        features = Dense(num_input_features)(key)
        for _ in range(num_layers):
            features = Dense(growth_rate)(features)
            features = BatchNorm()(features)
            features = Relu()(features)
            features = Conv2D(growth_rate, (3, 3), padding='SAME', use_bias=False)(features)
            features = Dropout(rate=0.0)(features)
        return features

    return init

batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (batch_size, num_input_features, height, width))
    return inputs

def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]
