import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GroupNormalization, Scale, MaxPooling2D, ClippedL2Normalization

class Model(tf.keras.Model):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding='same')
        self.group_norm = GroupNormalization(groups=num_groups)
        self.scale = Scale(scale_shape)
        self.maxpool = MaxPooling2D(pool_size=maxpool_kernel_size, strides=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.scale(x)
        x = self.maxpool(x)
        x = tf.clip_by_value(x, clip_value_min=self.clamp_min, clip_value_max=self.clamp_max)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
