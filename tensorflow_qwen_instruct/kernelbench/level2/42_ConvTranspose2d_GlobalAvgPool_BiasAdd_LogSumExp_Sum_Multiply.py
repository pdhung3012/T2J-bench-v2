import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, padding='same')
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv_transpose(x)
        x = tf.reduce_mean(x, axis=(1, 2), keepdims=True)  # Global average pooling
        x = x + self.bias
        x = tf.reduce_logsumexp(x, axis=1, keepdims=True)  # Log-sum-exp
        x = tf.reduce_sum(x, axis=(1, 2))  # Sum
        x = x * 10.0  # Multiplication
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
