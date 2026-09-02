import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Add, AveragePooling2D, Flatten, Dense

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = Conv2D(1280, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        
        # Fully connected layer
        self.fc = Dense(num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride of the depthwise convolution.
        :param expand_ratio: Expansion ratio for the hidden layer.
        :return: A sequential MBConv block.
        """
        hidden_dim = int(in_channels * expand_ratio)
        return tf.keras.Sequential([
            Conv2D(hidden_dim, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False),
            BatchNormalization(),
        ])
    
    def call(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = ReLU()(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = ReLU()(self.bn2(self.conv2(x)))
        x = AveragePooling2D(pool_size=(1, 1))(x)
        x = Flatten()(x)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, *input_shape))]

def get_init_inputs():
    return [num_classes]
