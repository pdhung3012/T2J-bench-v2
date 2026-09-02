import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation

class Model(tf.keras.Model):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')

    def call(self, inputs):
        """
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(inputs)
        x = tf.nn.hard_swish(x)
        x = tf.nn.relu(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
