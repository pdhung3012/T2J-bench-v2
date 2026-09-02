import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add

class Model(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(Model, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = tf.keras.Sequential([
                Conv2D(out_channels * self.expansion, kernel_size=1, strides=stride, padding='same', use_bias=False),
                BatchNormalization()
            ])
        self.stride = stride

    def call(self, inputs):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        out += identity
        out = self.relu(out)

        return out

# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, in_channels, 224, 224))]

def get_init_inputs():
    return [in_channels, out_channels, stride]
