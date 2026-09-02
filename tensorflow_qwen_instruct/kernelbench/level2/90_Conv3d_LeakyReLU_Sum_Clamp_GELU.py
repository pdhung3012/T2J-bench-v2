import tensorflow as tf
from tensorflow.keras.layers import Conv3D, LeakyReLU, Add, ClippedL2Normalization, GELU

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, input_shape=(None, depth, height, width))
        self.sum_tensor = tf.Variable(tf.random.normal(sum_tensor_shape))

    def call(self, x):
        x = self.conv(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = x + self.sum_tensor
        x = tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)
        x = GELU()(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
depth, height, width = 16, 64, 64
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
