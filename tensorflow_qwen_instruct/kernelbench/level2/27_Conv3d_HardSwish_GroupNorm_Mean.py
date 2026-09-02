import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Activation, GroupNormalization, GlobalAveragePooling1D

class Model(tf.keras.Model):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, padding='same', use_bias=bias)
        self.group_norm = GroupNormalization(groups=num_groups)

    def call(self, inputs):
        x = self.conv(inputs)                             # (B, C, D, H, W)
        x = Activation('hard_sigmoid')(x)                 # Nonlinear activation
        x = self.group_norm(x)                            # Normalization over channels
        x = tf.reduce_mean(x, axis=[2, 3, 4])             # Mean over spatial dims → (B, C)
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
