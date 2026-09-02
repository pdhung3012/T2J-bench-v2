import tensorflow as tf
from tensorflow.keras.layers import AvgPool3D, Conv3DTranspose, LayerNormalization, Softmax

class Model(tf.keras.Model):
    """
    Model that performs average pooling, 3D transposed convolution, clamping,
    spatial softmax, and multiplication by a learnable scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.avg_pool = AvgPool3D(pool_size=pool_kernel_size)
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = tf.Variable(tf.ones([1, out_channels, 1, 1, 1]))

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = tf.clip_by_value(x, clip_value_min=self.clamp_min, clip_value_max=self.clamp_max)
        b, c, d, h, w = x.shape
        x = tf.reshape(x, [b, c, -1])                     # flatten spatial dims
        x = Softmax(axis=2)(x)
        x = tf.reshape(x, [b, c, d, h, w])
        x = x * self.scale
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 32, 64, 64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]
