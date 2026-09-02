import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Softmax

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), padding='same')
        self.pool1 = MaxPooling3D(pool_size=(pool_kernel_size, pool_kernel_size, pool_kernel_size))
        self.pool2 = MaxPooling3D(pool_size=(pool_kernel_size, pool_kernel_size, pool_kernel_size))

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        x = tf.nn.softmax(x, axis=1)
        x = self.pool1(x)
        x = self.pool2(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
