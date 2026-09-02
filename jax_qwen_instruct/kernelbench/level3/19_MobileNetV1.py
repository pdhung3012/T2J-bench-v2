import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, Dense, AvgPool

def conv_bn(inp, oup, stride):
    return Conv(inp, oup, 3, stride, 1, has_bias=False), BatchNorm(oup), Relu()

def conv_dw(inp, oup, stride):
    return (
        Conv(inp, inp, 3, stride, 1, groups=inp, has_bias=False),
        BatchNorm(inp),
        Relu(),
        Conv(inp, oup, 1, 1, 0, has_bias=False),
        BatchNorm(oup),
        Relu(),
    )

net = [
    conv_bn(3, int(32 * alpha), 2),
    conv_dw(int(32 * alpha), int(64 * alpha), 1),
    conv_dw(int(64 * alpha), int(128 * alpha), 2),
    conv_dw(int(128 * alpha), int(128 * alpha), 1),
    conv_dw(int(128 * alpha), int(256 * alpha), 2),
    conv_dw(int(256 * alpha), int(256 * alpha), 1),
    conv_dw(int(256 * alpha), int(512 * alpha), 2),
    conv_dw(int(512 * alpha), int(512 * alpha), 1),
    conv_dw(int(512 * alpha), int(512 * alpha), 1),
    conv_dw(int(512 * alpha), int(512 * alpha), 1),
    conv_dw(int(512 * alpha), int(512 * alpha), 1),
    conv_dw(int(512 * alpha), int(1024 * alpha), 2),
    conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
    AvgPool(7),
]

def model(inputs, params):
    for layer in net[:-1]:
        inputs = layer[0](inputs, params) + layer[1](inputs, params) + layer[2](inputs, params)
    inputs = AvgPool(7)(inputs, params)
    inputs = Dense(1024 * alpha, name='fc')(inputs, params)
    return Dense(num_classes, name='fc_out')(inputs, params)

def get_inputs():
    return [jnp.random.rand(*shape).astype(jnp.float32) for shape in get_shape()]

def get_init_inputs():
    return [num_classes, 3, alpha]
