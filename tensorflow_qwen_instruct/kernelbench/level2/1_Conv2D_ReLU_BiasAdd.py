import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv(x)
        x = ReLU()(x)
        x = x + self.bias
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
