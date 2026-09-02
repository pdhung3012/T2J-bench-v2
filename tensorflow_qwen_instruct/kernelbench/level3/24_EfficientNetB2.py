import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Dense

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Define the EfficientNetB2 architecture components
        self.conv1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        
        # Define the MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers
        self.conv_final = Conv2D(1408, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)
        self.bn_final = BatchNormalization()
        self.avgpool = AveragePooling2D(pool_size=(1, 1))
        self.fc = Dense(num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the MBConv block.
        :return: A sequential container of layers forming the MBConv block.
        """
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(Conv2D(expanded_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
            layers.append(BatchNormalization())
            layers.append(ReLU())
        
        # Depthwise convolution
        layers.append(Conv2D(expanded_channels, kernel_size=(3, 3), strides=stride, padding='same', use_bias=False, groups=expanded_channels))
        layers.append(BatchNormalization())
        layers.append(ReLU())
        
        # Squeeze and Excitation
        layers.append(AveragePooling2D(pool_size=(1, 1)))
        layers.append(Conv2D(expanded_channels // 4, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
        layers.append(ReLU())
        layers.append(Conv2D(expanded_channels // 4, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
        layers.append(ReLU())
        layers.append(ReLU())
        
        # Output phase
        layers.append(Conv2D(expanded_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False))
        layers.append(BatchNormalization())
        
        return tf.keras.Sequential(layers)
    
    def call(self, x):
        """
        Forward pass of the EfficientNetB2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)
        return x

# Test code
batch_size = 2
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
