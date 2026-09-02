import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, Dense, Flatten, MaxPool, Concatenate, Initializer

def model(in_channels, out_channels, stride=1):
    net = [
        Conv(in_channels, out_channels, (3, 3), stride=(stride, stride), padding=(1, 1)),
        BatchNorm(),
        Relu(),
        Conv(out_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
        BatchNorm(),
    ]
    
    if stride != 1:
        shortcut = [
            Conv(in_channels, out_channels * 4, (1, 1), stride=(stride, stride)),
            BatchNorm()
        ]
    else:
        shortcut = [Identity()]
    
    net += shortcut
    net += [Relu()]
    
    return net

def forward(params, state, inputs):
    out = inputs
    for layer in model(params['in_channels'], params['out_channels']):
        out = layer(out, **params)
    return out

def loss(params, state, inputs, targets):
    logits = forward(params, state, inputs)
    loss_value = jnp.mean(jnp.square(logits - targets))
    return loss_value

def main():
    in_channels = 3
    out_channels = 64
    stride = 1
    batch_size = 10
    num_classes = 1000

    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, 224, 224))
    targets = jax.random.randint(jax.random.PRNGKey(1), (batch_size,), low=0, high=num_classes)

    init_params = {
        'in_channels': in_channels,
        'out_channels': out_channels
    }

    optimizer_init, optimizer_update, get_params = optimizers.adam(learning_rate=0.001)
    optimizer_state = optimizer_init(init_params)
    params = get_params(optimizer_state)

    for _ in range(100):
        grads = jax.grad(loss)(params, optimizer_state, inputs, targets)
        optimizer_state = optimizer_update(optimizer_state, grads, optimizer_state)

    print("Final Loss:", loss(params, optimizer_state, inputs, targets))

if __name__ == "__main__":
    main()
