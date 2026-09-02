import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Conv, BatchNorm, Relu, ShuffleChannels, Add, Sequential

def Model(in_channels, out_channels, groups=3):
    """
    ShuffleNet unit implementation.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param groups: Number of groups for group convolution.
    """
    assert out_channels % 4 == 0
    mid_channels = out_channels // 4
    
    net = [
        Dense(mid_channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False),
        BatchNorm(),
        Conv(mid_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME', groups=mid_channels, use_bias=False),
        BatchNorm(),
        Dense(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME', use_bias=False),
        BatchNorm(),
        ShuffleChannels(groups),
        Relu(),
        Add(),
    ]
    
    return Sequential(net)

def ChannelShuffle(groups):
    """
    Channel shuffle operation.

    :param groups: Number of groups for shuffling.
    """
    return ShuffleChannels(groups)

batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_channels, height, width))]

def get_init_inputs():
    return [input_channels, out_channels, groups]
