import tensorflow as tf
from tensorflow.keras.layers import Conv3D, GroupNormalization

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, padding='same')
        self.group_norm = GroupNormalization(groups=num_groups)

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = tf.reduce_mean(x, axis=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
