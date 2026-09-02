import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, LogSumExp, ReLU

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), strides=stride, padding='same')
        self.max_pool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = tf.reduce_logsumexp(x, axis=1, keepdims=True)
        x = ReLU()(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
