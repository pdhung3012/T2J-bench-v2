import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, LeakyReLU, MaxPooling3D, LayerNormalization

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same')
        self.multiplier = tf.Variable(tf.random.normal(multiplier_shape))
        self.leaky_relu = LeakyReLU(alpha=0.2)
        self.max_pool = MaxPooling3D(pool_size=(1, 2, 2))

    def call(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
