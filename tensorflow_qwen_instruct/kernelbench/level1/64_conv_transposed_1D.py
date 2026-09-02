import tensorflow as tf
from tensorflow.keras.layers import Conv1DTranspose

class Model(tf.keras.Model):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = Conv1DTranspose(out_channels, kernel_size, strides=stride, padding='same', output_padding=output_padding, groups=groups, use_bias=bias)
        
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
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
# much larger signal length for heavier workload
length = 65536

def get_inputs():
    x = tf.random.normal((batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
