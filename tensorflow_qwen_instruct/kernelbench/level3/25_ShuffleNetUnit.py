import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, ShuffleChannel

class Model(tf.keras.Model):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(Model, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = Conv2D(mid_channels, kernel_size=1, strides=1, padding='same', groups=groups, use_bias=False)
        self.bn1 = BatchNormalization()
        
        # Depthwise 3x3 convolution
        self.conv2 = Conv2D(mid_channels, kernel_size=3, strides=1, padding='same', groups=mid_channels, use_bias=False)
        self.bn2 = BatchNormalization()
        
        # Second 1x1 group convolution
        self.conv3 = Conv2D(out_channels, kernel_size=1, strides=1, padding='same', groups=groups, use_bias=False)
        self.bn3 = BatchNormalization()
        
        # Shuffle operation
        self.shuffle = ShuffleChannel(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = tf.keras.Sequential()
        else:
            self.shortcut = tf.keras.Sequential([
                Conv2D(in_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
                BatchNormalization()
            ])
    
    def call(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = ReLU(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ShuffleChannel(tf.keras.layers.Layer):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ShuffleChannel, self).__init__()
        self.groups = groups
    
    def call(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // self.groups
        
        # Reshape
        x = tf.reshape(x, (batch_size, self.groups, channels_per_group, height, width))
        
        # Transpose
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        
        # Flatten
        x = tf.reshape(x, (batch_size, -1, height, width))
        
        return x

batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, input_channels, height, width))]

def get_init_inputs():
    return [input_channels, out_channels, groups]
