import tensorflow as tf
from tensorflow.keras.layers import Conv1DTranspose

class Model(tf.keras.Model):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

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
        self.conv1d_transpose = Conv1DTranspose(out_channels, kernel_size, strides=stride, padding='same', dilation_rate=dilation, use_bias=bias)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(inputs)

# Test code
batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 'same'
dilation = 3

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
