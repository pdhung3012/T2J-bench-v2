import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Dense

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding='same', output_padding=output_padding)
        self.bias = Dense(bias_shape[0], use_bias=False)(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv_transpose(x)
        x = x - self.bias
        x = tf.tanh(x)
        return x

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
