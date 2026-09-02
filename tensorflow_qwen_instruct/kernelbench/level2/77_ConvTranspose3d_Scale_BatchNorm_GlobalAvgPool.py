import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization, GlobalAveragePooling3D

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, activation=None)
        self.scale_factor = scale_factor
        self.batch_norm = BatchNormalization(epsilon=eps, momentum=momentum)
        self.global_avg_pool = GlobalAveragePooling3D()

    def call(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
