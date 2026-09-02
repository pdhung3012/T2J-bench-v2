import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, LayerNormalization
from tensorflow.keras.activations import gelu

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(in_channels, out_channels, kernel_size, strides=stride, padding=padding, use_bias=bias)
        self.layer_norm = LayerNormalization(epsilon=eps)
        self.scaling_factor = scaling_factor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        x = gelu(x)
        x = x * self.scaling_factor
        return x

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
