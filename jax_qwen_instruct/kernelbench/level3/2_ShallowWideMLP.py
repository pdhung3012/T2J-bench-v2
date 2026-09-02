import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Relu, Initializer, BatchNorm, Flatten

def model(input_size, hidden_layer_sizes, output_size):
    init_key = jax.random.PRNGKey(0)
    net = [
        Dense(hidden_layer_sizes[0], key=init_key, w_init=Initializer(jnp.ones), b_init=Initializer(jnp.zeros)),
        Relu(),
        *[
            Dense(hidden_layer_sizes[i+1], key=jax.random.fold_in(init_key, i), w_init=Initializer(jnp.ones), b_init=Initializer(jnp.zeros)) 
            for i in range(len(hidden_layer_sizes) - 1)
        ],
        Dense(output_size, key=jax.random.fold_in(init_key, len(hidden_layer_sizes)), w_init=Initializer(jnp.ones), b_init=Initializer(jnp.zeros))
    ]
    return net

def forward(params, inputs):
    activations = Flatten()(inputs)
    for layer in model(input_size, hidden_layer_sizes, output_size):
        activations = layer(activations)
    return activations

batch_size = 128
input_size = 16384
hidden_layer_sizes = [32768, 32768]
output_size = 16384

get_inputs = lambda: jax.random.normal(jax.random.PRNGKey(1), (batch_size, input_size))
get_init_inputs = lambda: (input_size, hidden_layer_sizes, output_size)

model_params = model(input_size, hidden_layer_sizes, output_size)
forward(model_params, get_inputs())
