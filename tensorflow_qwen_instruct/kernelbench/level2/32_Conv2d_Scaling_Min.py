import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(in_channels, height, width))
        self.scale_factor = scale_factor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = x * self.scale_factor
        x = tf.reduce_min(x, axis=1, keepdims=True)  # Minimum along channel dimension
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
