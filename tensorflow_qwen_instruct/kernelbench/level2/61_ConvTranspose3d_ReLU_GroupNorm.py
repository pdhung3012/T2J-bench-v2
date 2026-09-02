import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, ReLU, GroupNormalization

class Model(tf.keras.Model):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, padding='same', use_bias=bias)
        self.relu = ReLU()
        self.group_norm = GroupNormalization(groups=groups)

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
