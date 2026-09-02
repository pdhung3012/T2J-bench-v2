import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, MaxPooling2D, HardSigmoid, Mean, Tanh

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride,), padding='same')
        self.maxpool = MaxPooling2D(pool_size=maxpool_kernel_size, strides=maxpool_stride)
        self.hardtanh = HardSigmoid(min=hardtanh_min, max=hardtanh_max)

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        x = Tanh()(x)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
