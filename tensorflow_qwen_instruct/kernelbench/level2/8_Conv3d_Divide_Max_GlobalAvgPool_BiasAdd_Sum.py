import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AdaptiveAveragePooling3D, Dense

class Model(tf.keras.Model):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = Conv3D(out_channels, kernel_size, input_shape=(None, depth, height, width))
        self.divisor = divisor
        self.max_pool = MaxPooling3D(pool_size)
        self.global_avg_pool = AdaptiveAveragePooling3D((1, 1, 1))
        self.bias = Dense(bias_shape[-1], use_bias=False, input_shape=bias_shape)
        self.sum_dim = sum_dim

    def call(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.bias(x)
        x = tf.reduce_sum(x, axis=self.sum_dim)
        return x

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, depth, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
