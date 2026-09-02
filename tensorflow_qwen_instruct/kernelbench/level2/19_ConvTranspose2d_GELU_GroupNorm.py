import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, GroupNormalization, Gelu

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding='same')
        self.group_norm = GroupNormalization(groups=num_groups, axis=-1)

    def call(self, x):
        x = self.conv_transpose(x)
        x = Gelu()(x)
        x = self.group_norm(x)
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
