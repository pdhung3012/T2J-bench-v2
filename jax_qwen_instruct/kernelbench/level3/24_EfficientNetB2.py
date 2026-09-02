import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, AvgPool, Dense, AdaptiveAvgPool2d, AdapMaxPool2d, Sigmoid

class Model:
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        net = [
            Conv(32, (3, 3), strides=(2, 2), padding=(1, 1), use_bias=False),
            BatchNorm(),
            Relu(),
            self._make_mbconv_block(32, 96, 1, 3),
            self._make_mbconv_block(96, 144, 2, 6),
            self._make_mbconv_block(144, 192, 2, 6),
            self._make_mbconv_block(192, 288, 2, 6),
            self._make_mbconv_block(288, 384, 1, 6),
            Conv(384, (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False),
            BatchNorm(),
            AdaptiveAvgPool2d((1, 1)),
            Dense(1408, use_bias=False),
            BatchNorm(),
            Dense(num_classes)
        ]
        self.net = jax.experimental.stax.serial(*net)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the MBConv block.
        :return: A sequential container of layers forming the MBConv block.
        """
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            net = [
                Conv(expanded_channels, (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False),
                BatchNorm(),
                Relu(),
            ]
        else:
            net = []

        # Depthwise convolution
        net += [
            Conv(expanded_channels, (3, 3), strides=stride, padding=(1, 1), groups=expanded_channels, use_bias=False),
            BatchNorm(),
            Relu(),
        ]

        # Squeeze and Excitation
        net += [
            AdaptiveAvgPool2d((1, 1)),
            Conv(expanded_channels // 4, (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False),
            Relu(),
            Conv(expanded_channels // 4, (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False),
            Sigmoid(),
        ]

        # Output phase
        net += [
            Conv(out_channels, (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False),
            BatchNorm(),
        ]
        return jax.experimental.stax.serial(*net)

    @jax.jit
    def forward(self, x):
        """
        Forward pass of the EfficientNetB2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = Relu()(BatchNorm()(Conv((32, 32), (3, 3), strides=(2, 2), padding=(1, 1), use_bias=False)(x)))
        x = self.net(x)
        x = Relu()(BatchNorm()(Conv((384, 384), (1, 1), strides=(1, 1), padding=(0, 0), use_bias=False)(x)))
        x = AdaptiveAvgPool2d((1, 1))(x)
        x = Dense(1408, use_bias=False)(x)
        x = Relu()(BatchNorm()(x))
        x = Dense(self.num_classes)(x)
        return x

# Test code
batch_size = 2
num_classes = 1000

def get_inputs():
    return [jnp.random.rand(batch_size, 3, 224, 224).astype(jnp.float32)]

def get_init_inputs():
    return [num_classes]
