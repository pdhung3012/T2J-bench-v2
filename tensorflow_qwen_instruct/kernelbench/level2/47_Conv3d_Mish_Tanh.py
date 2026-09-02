import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Mish, Activation

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, strides=stride, padding='same')

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = Mish()(x)
        x = Activation('tanh')(x)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, D, H, W])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
