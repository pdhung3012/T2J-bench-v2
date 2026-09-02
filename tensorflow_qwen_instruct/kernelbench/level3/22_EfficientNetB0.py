import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Dense

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in TensorFlow.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        
        # MBConv blocks
        self.blocks = tf.keras.Sequential([
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, strides=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, strides=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, strides=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, strides=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, strides=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, strides=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, strides=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, strides=1, expand_ratio=6)
        ])
        
        # Final convolutional layer
        self.conv2 = Conv2D(1280, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        
        # Fully connected layer
        self.fc = Dense(num_classes)
    
    def call(self, inputs):
        """
        Forward pass of the EfficientNetB0 model.

        :param inputs: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = ReLU()(self.bn1(self.conv1(inputs)))
        x = self.blocks(x)
        x = ReLU()(self.bn2(self.conv2(x)))
        x = AveragePooling2D((1, 1))(x)
        x = Flatten()(x)
        x = self.fc(x)
        return x

class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = tf.keras.Sequential([
                Conv2D(hidden_dim, kernel_size=1, strides=1, padding='same', use_bias=False),
                BatchNormalization(),
                ReLU()
            ])
        
        self.depthwise_conv = tf.keras.Sequential([
            Conv2D(hidden_dim, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False, groups=hidden_dim),
            BatchNormalization(),
            ReLU()
        ])
        
        self.project_conv = tf.keras.Sequential([
            Conv2D(out_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            BatchNormalization()
        ])
    
    def call(self, inputs):
        """
        Forward pass of the MBConv block.

        :param inputs: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = inputs
        
        if hasattr(self, 'expand_conv'):
            inputs = self.expand_conv(inputs)
        
        inputs = self.depthwise_conv(inputs)
        inputs = self.project_conv(inputs)
        
        if self.use_residual:
            inputs += identity
        
        return inputs

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [tf.random.normal([batch_size, 3, 224, 224])]

def get_init_inputs():
    return [num_classes]
