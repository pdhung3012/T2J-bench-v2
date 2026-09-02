import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the pointwise 2D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv1d(inputs)

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
