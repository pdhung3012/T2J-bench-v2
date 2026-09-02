import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, LogSumExp, Activation, Subtract, ClippedL2Normalization

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same')
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv_transpose(x)
        x = LogSumExp()(x, axis=1, keepdims=True)
        x = x * Activation('sigmoid')(x + 3) / 6
        x = x - self.bias[tf.newaxis, tf.newaxis, tf.newaxis, ...]
        x = ClippedL2Normalization()(x, max_value=1.0, min_value=-1.0)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
