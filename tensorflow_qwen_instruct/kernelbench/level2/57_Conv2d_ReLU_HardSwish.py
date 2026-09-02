import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Multiply, Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, input_shape=(height, width, in_channels))

    def call(self, x):
        x = self.conv(x)
        x = ReLU()(x)
        x = Multiply()([x, Activation('hard_sigmoid')(x + 3) / 6])
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
