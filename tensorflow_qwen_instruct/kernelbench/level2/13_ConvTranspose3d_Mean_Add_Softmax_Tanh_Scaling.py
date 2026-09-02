import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization, Activation, Multiply

class Model(tf.keras.Model):
    """
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling (across depth)
    3. Addition
    4. Softmax (across channels)
    5. Tanh activation
    6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same')
        self.bias = tf.Variable(tf.random.normal([1, out_channels, 1, 1, 1]))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def call(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = tf.reduce_mean(x, axis=2, keepdims=True)          # Mean pool over depth dim (D)
        x = x + self.bias                                     # Bias add per channel
        x = tf.nn.softmax(x, axis=1)                          # Softmax over channels
        x = tf.nn.tanh(x)                                     # Nonlinearity
        x = x * self.scaling_factor                           # Scaling
        return x

# === Test config ===
batch_size = 16
in_channels  = 16  
out_channels = 64  
depth = 32; height = width = 128  
kernel_size  = 3
stride       = 1  
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]
