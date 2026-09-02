import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Relu, Dense, Concatenate

in_channels = 3
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

model = lambda: [
    Conv(squeeze_channels, (1), padding=0),
    Relu(),
    Conv(expand1x1_channels, (1), padding=0),
    Relu(),
    Conv(expand3x3_channels, (3, 3), padding=1),
    Relu(),
    Concatenate(axis=1)
]

def model_fn(params, inputs):
    return model()(inputs, params)

def loss_fn(params, batch):
    logits = model_fn(params, batch)
    y = jnp.ones((logits.shape[0], 1))
    return jnp.mean(jnp.square(logits - y))

def main():
    key = jax.random.PRNGKey(0)
    init_params = model()[-1](key, jnp.zeros((1, in_channels, height, width)))
    optimizer_init, optimizer_update = optimizers.adam(step_size=0.01)
    optimizer_state = optimizer_init(init_params)
    _, optimizer_state = jax.grad(loss_fn)(optimizer_state, get_inputs())

if __name__ == "__main__":
    main()
