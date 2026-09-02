import tensorflow as tf
from tensorflow.keras.layers import Conv3D, InstanceNormalization, LeakyReLU, Multiply, MaxPool3D

class Model(tf.keras.Model):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, padding='same')
        self.multiplier = tf.Variable(tf.random.normal(multiplier_shape))
        self.instance_norm = InstanceNormalization(axis=-1)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def call(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = tf.clip_by_value(x, clip_value_min=self.clamp_min, clip_value_max=self.clamp_max)
        x = x * self.multiplier
        x = tf.reduce_max(x, axis=1)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
