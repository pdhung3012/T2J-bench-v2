import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, AvgPool, Dense, Flatten, MaxPool, Serial, Permute

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = jax.lax.ceil(v / divisor) * divisor
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_residual_block(inp, oup, stride, expand_ratio):
    hidden_dim = _make_divisible(inp * expand_ratio, 8)
    use_res_connect = stride == 1 and inp == oup

    layers = []
    if expand_ratio != 1:
        layers.append(Conv(inp, hidden_dim, kernel=1, padding=0))
        layers.append(BatchNorm(hidden_dim))
        layers.append(Relu())

    layers.extend([
        Conv(hidden_dim, hidden_dim, kernel=3, padding=1, stride=stride, use_bias=False),
        BatchNorm(hidden_dim),
        Relu(),
        Conv(hidden_dim, oup, kernel=1, padding=0, use_bias=False),
        BatchNorm(oup),
    ])

    if use_res_connect:
        return Serial(layers), True
    else:
        return Serial(layers), False

input_channel = 32
last_channel = 1280
inverted_residual_setting = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

features = [
    Conv(3, input_channel, kernel=3, padding=1),
    BatchNorm(input_channel),
    Relu(),
]

for t, c, n, s in inverted_residual_setting:
    output_channel = _make_divisible(c, 8)
    for _ in range(n):
        stride = s if _ == 0 else 1
        block, shortcut = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
        features.extend(block)
        input_channel = output_channel

features.extend([
    Conv(input_channel, last_channel, kernel=1),
    BatchNorm(last_channel),
    Relu(),
    AvgPool(1),
])

classifier = Dense(last_channel, num_classes)

model = Serial(Flatten, *features, classifier)

def forward(params, state, inputs):
    x = inputs['image']
    for layer in model:
        x = layer(x, params=params)
    return {'logits': x}

def get_inputs():
    return {'image': jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3, 224, 224))}

def get_init_inputs():
    return [num_classes]
