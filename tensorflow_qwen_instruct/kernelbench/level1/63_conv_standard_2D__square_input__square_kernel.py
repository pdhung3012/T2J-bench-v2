import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = Conv2D(out_channels, (kernel_size, kernel_size), strides=(stride, stride), padding='same', dilation_rate=(dilation, dilation), groups=groups, use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the 2D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(inputs)

# Test code
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
