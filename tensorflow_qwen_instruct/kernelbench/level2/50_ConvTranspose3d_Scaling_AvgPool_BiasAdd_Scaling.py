import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, AveragePooling3D

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding)
        self.scale1 = tf.Variable(scale1)
        self.avg_pool = AveragePooling3D(pool_size=2)
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.scale2 = tf.Variable(scale2)

    def call(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = x + self.bias
        x = x * self.scale2
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
