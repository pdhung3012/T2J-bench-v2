import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Tanh, MaxPooling2D, GroupNormalization

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding='same')
        self.batch_norm = BatchNormalization()
        self.tanh = Tanh()
        self.max_pool = MaxPooling2D(pool_size=2, strides=2)
        self.group_norm = GroupNormalization(groups=num_groups)

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
