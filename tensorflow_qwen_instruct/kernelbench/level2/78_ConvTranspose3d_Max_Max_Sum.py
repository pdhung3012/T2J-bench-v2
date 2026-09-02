import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, MaxPooling3D

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(stride, stride, stride), padding='same')
        self.max_pool1 = MaxPooling3D(pool_size=(2, 2, 2))
        self.max_pool2 = MaxPooling3D(pool_size=(3, 3, 3))

    def call(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = tf.reduce_sum(x, axis=1, keepdims=True)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
