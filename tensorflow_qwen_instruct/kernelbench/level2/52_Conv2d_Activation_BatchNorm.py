import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')
        self.bn = BatchNormalization(epsilon=eps, momentum=momentum)

    def call(self, x):
        x = self.conv(x)
        x = tf.multiply(tf.tanh(tf.nn.softplus(x)), x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
