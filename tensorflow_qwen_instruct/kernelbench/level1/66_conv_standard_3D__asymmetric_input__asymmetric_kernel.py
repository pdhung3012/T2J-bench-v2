import tensorflow as tf
from tensorflow.keras.layers import Conv3D

class Model(tf.keras.Model):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = Conv3D(out_channels, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, groups=groups, use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the 3D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv3d(inputs)

# Test code
batch_size = 8
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 128
width = 128

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
