import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation

class Model(tf.keras.Model):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size=(kernel_size, kernel_size), padding='same')
        self.scaling_factor = scaling_factor
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.max_pool = MaxPooling2D(pool_size=(pool_kernel_size, pool_kernel_size))

    def call(self, inputs):
        # Convolution
        x = self.conv(inputs)
        # Tanh activation
        x = tf.nn.tanh(x)
        # Scaling
        x = x * self.scaling_factor
        # Bias addition
        x = x + self.bias
        # Max-pooling
        x = self.max_pool(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
