import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, LayerNormalization, AveragePooling3D, GELU

class Model(tf.keras.Model):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = Conv3DTranspose(out_channels, kernel_size, strides=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = tf.Variable(initial_value=sum_weight, dtype=tf.float32)
        self.norm = LayerNormalization(norm_shape)
        self.avg_pool = AveragePooling3D(pool_size=pool_kernel_size)
        self.gelu = GELU()

    def call(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
