import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, MaxPooling3D, Softmax, Subtract, Swish, Max

class Model(tf.keras.Model):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3D
        - MaxPool3D
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=(1, stride, stride), padding='same', output_padding=output_padding)
        self.max_pool = MaxPooling3D(pool_size=(1, pool_kernel_size, pool_kernel_size), strides=(1, pool_stride, pool_stride), padding='same')
        self.subtract = Subtract()
        self.swish = Swish()
        self.max = Max()

    def call(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = Softmax(axis=1)(x) # Apply softmax across channels (axis=1)
        x = x - self.subtract([x]) # Subtract across channels
        x = self.swish(x) * x # Swish activation
        x = self.max(x) # Max pooling across channels
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, depth, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]
