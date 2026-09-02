import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

class Model(tf.keras.Model):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(Model, self).__init__()
        
        self.squeeze = Conv2D(squeeze_channels, kernel_size=1, padding='same')
        self.squeeze_activation = ReLU()
        
        self.expand1x1 = Conv2D(expand1x1_channels, kernel_size=1, padding='same')
        self.expand1x1_activation = ReLU()
        
        self.expand3x3 = Conv2D(expand3x3_channels, kernel_size=3, padding='same')
        self.expand3x3_activation = ReLU()
    
    def call(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        x = self.squeeze_activation(self.squeeze(x))
        return tf.concat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], axis=1)

# Test code
batch_size = 128
num_input_features = 3
num_output_features = 64
height, width = 256, 256
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [tf.random.normal((batch_size, num_input_features, height, width))]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
