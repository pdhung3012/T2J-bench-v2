import tensorflow as tf
from tensorflow.keras.layers import Conv3D, GlobalMaxPooling3D, Softmax

class Model(tf.keras.Model):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension,
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, input_shape=(in_channels, D, H, W))
        self.dim = dim

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = tf.reduce_min(x, axis=self.dim)  # Apply minimum along the specified dimension
        x = Softmax(axis=1)(x)  # Apply softmax along the channel dimension
        return x

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
