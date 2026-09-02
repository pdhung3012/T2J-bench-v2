import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose

class Model(tf.keras.Model):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = Conv3DTranspose(out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), strides=(stride, stride, stride), padding='same', groups=groups, use_bias=bias)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(inputs)

# Test code
batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
