import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D

class Model(tf.keras.Model):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size=(kernel_size, kernel_size), input_shape=(in_channels, height, width))
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = AveragePooling2D(pool_size=kernel_size_pool)

    def call(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = tf.nn.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
