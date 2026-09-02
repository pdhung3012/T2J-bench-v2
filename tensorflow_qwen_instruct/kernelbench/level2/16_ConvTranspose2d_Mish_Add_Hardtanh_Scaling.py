import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Mish, HardSigmoid

class Model(tf.keras.Model):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(Model, self).__init__()
        self.conv_transpose = Conv2DTranspose(out_channels, kernel_size, strides=(stride, stride), padding='same', output_padding=output_padding)
        self.add_value = add_value
        self.scale = scale

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = Mish()(x) # Mish activation
        x = x + self.add_value
        x = HardSigmoid()(x) # Hardtanh activation
        x = x * self.scale # Scaling
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, height, width])]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
