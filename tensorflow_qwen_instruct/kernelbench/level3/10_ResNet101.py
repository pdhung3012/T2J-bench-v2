import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, AveragePooling2D, Dense, Input, Sequential

class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(out_channels * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels * expansion, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Model(tf.keras.Model):
    def __init__(self, layers, num_classes=1000):
        """
        :param block: Type of block to use (BasicBlock or Bottleneck)
        :param layers: List of integers specifying the number of blocks in each layer
        :param num_classes: Number of output classes
        """
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=3, strides=2, padding='same')

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = AveragePooling2D(pool_size=(7, 7))
        self.fc = Dense(num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = Conv2D(out_channels * block.expansion, kernel_size=1, strides=stride, use_bias=False)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return tf.keras.Sequential(layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
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
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, 3, height, width))]

def get_init_inputs():
    return [layers, num_classes]
