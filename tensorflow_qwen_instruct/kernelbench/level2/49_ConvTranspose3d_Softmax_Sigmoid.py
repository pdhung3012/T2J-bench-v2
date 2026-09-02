import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, Softmax, Sigmoid

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding, output_padding=output_padding, use_bias=bias)
        self.softmax = Softmax(axis=1)
        self.sigmoid = Sigmoid()

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.softmax(x)
        x = self.sigmoid(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, D, H, W])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
