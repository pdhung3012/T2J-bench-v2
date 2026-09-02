import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.depthwise = Conv2D(in_channels, kernel_size, strides=stride, padding='same', dilation_rate=dilation, groups=in_channels, use_bias=bias)
        self.pointwise = Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return x

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
