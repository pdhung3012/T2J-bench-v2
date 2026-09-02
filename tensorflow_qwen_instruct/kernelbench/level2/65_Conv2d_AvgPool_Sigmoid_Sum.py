import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Sigmoid

class Model(tf.keras.Model):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size=(kernel_size, kernel_size), padding='same')
        self.avg_pool = AveragePooling2D(pool_size=(pool_kernel_size, pool_kernel_size))

    def call(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = Sigmoid()(x)
        x = tf.reduce_sum(x, axis=[1, 2, 3])  # Sum over all spatial dimensions
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
