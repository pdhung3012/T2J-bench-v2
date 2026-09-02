import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(in_channels, height, width))

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = tf.nn.gelu(x)
        x = GlobalAveragePooling2D()(x)
        x = tf.squeeze(x, -1)
        x = tf.squeeze(x, -1)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
