import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, LayerNormalization, Add

class Model(tf.keras.Model):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride, stride), padding='same', output_padding=output_padding)
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        x = self.conv_transpose(x)
        x = tf.reduce_min(x, axis=1, keepdims=True)  # Minimum operation along channel dimension
        x = tf.reduce_sum(x, axis=2, keepdims=True)  # Sum operation along height dimension
        x = tf.nn.gelu(x)  # GELU activation
        x = x + self.bias
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
