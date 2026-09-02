import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Initializer, BatchNorm, Flatten

def model(input_size, layer_sizes, output_size):
    init_layers = []
    current_input_size = input_size
    
    for layer_size in layer_sizes:
        init_layers.append(Dense(layer_size, key=jax.random.PRNGKey(current_input_size), w_init=Initializer(jnp.ones), b_init=Initializer(jnp.zeros)))
        init_layers.append(Relu())
        current_input_size = layer_size
    
    init_layers.append(Dense(output_size, key=jax.random.PRNGKey(current_input_size), w_init=Initializer(jnp.ones), b_init=Initializer(jnp.zeros)))
    
    return init_layers

def forward(params, inputs):
    net = params
    x = inputs[0]
    
    for layer in net[:-1]:
        x = layer(x)
    
    return net[-1](x)

batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
