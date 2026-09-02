import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, AveragePooling2D, Flatten, Dense

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(Model, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return tf.keras.Sequential([
                Conv2D(oup, 3, strides=stride, padding='same', use_bias=False),
                BatchNormalization(),
                Activation('relu')
            ])
        
        def conv_dw(inp, oup, stride):
            return tf.keras.Sequential([
                DepthwiseConv2D(oup, kernel_size=3, strides=stride, padding='same', use_bias=False),
                BatchNormalization(),
                Activation('relu'),
                
                Conv2D(oup, 1, strides=1, padding='same', use_bias=False),
                BatchNormalization(),
                Activation('relu'),
            ])
        
        self.model = tf.keras.Sequential([
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            AveragePooling2D(pool_size=7),
        ])
        self.fc = tf.keras.layers.Dense(num_classes)
    
    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.model(inputs, training=training)
        x = Flatten()(x)
        x = self.fc(x)
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, input_channels, height, width))]

def get_init_inputs():
    return [num_classes, input_channels, alpha]
