import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, MaxPooling3D, AdaptiveAveragePooling3D, ClippedActivation

class Model(tf.keras.Model):
    """
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same')
        self.scale = scale
        self.maxpool = MaxPooling3D(pool_size=(1, maxpool_kernel_size, maxpool_kernel_size), strides=(1, 2, 2))
        self.global_avg_pool = AdaptiveAveragePooling3D((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = x * self.scale
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = ClippedActivation(min=self.clamp_min, max=self.clamp_max)(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
