import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = Conv2D(in_channels, kernel_size, strides=stride, padding='same', groups=in_channels, use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.conv2d(inputs)

# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
