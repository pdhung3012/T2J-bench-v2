import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add

class Model(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = tf.keras.Sequential([
                Conv2D(hidden_dim, kernel_size=1, strides=1, padding='same', use_bias=False),
                BatchNormalization(),
                ReLU(6, activation_mode='linear')
            ])
        
        self.depthwise_conv = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False),
            BatchNormalization()
        ])
        
        self.project_conv = tf.keras.Sequential([
            Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            BatchNormalization()
        ])
    
    def call(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, 224, 224))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
