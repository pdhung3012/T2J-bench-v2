import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, AveragePooling2D, Dense, Layer
from tensorflow.keras.models import Model

class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs):
        """
        :param inputs: Input tensor, shape (batch_size, in_channels, height, width)
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

class Model(Model):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = AveragePooling2D(pool_size=(7, 7), strides=1)
        self.fc = Dense(num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = Conv2D(out_channels * block.expansion, kernel_size=1, strides=stride, padding='same', use_bias=False)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return layers

    def call(self, inputs):
        """
        :param inputs: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)

        return x

# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [tf.random.normal(input_shape)]

def get_init_inputs():
    return [num_classes]
