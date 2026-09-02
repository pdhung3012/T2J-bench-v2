import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, Reshape

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, followed by a sum,
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same', output_padding=output_padding)
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv_transpose(x)
        original_x = x.numpy().copy()
        x = x + self.bias
        x = x + original_x
        x = x * original_x
        x = x + original_x
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
