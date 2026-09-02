import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization, AveragePooling3D

class Model(tf.keras.Model):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding)
        self.batch_norm = BatchNormalization()
        self.avg_pool1 = AveragePooling3D(pool_size=2)
        self.avg_pool2 = AveragePooling3D(pool_size=2)

    def call(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x

batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
