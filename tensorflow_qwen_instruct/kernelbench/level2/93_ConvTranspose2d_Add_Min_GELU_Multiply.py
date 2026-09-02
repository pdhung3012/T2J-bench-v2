import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Add, Min, Gelu, Multiply

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(1, stride), padding='same')
        self.add_value = add_value
        self.multiply_value = multiply_value

    def call(self, x):
        x = self.conv_transpose(x)
        x = x + self.add_value
        x = Min()(x, tf.constant(0.0, shape=x.shape, dtype=x.dtype))
        x = Gelu()(x)
        x = x * self.multiply_value
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
