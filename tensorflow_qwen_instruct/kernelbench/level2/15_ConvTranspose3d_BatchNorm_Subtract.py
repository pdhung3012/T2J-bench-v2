import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization

class Model(tf.keras.Model):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding, use_bias=bias)
        self.batch_norm = BatchNormalization()

    def call(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x - tf.reduce_mean(x, axis=[2, 3, 4], keepdims=True)  # Subtract mean along spatial dimensions
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
