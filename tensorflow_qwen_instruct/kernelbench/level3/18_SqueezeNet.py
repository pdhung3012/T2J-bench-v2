import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, Dropout, AdaptiveAveragePooling2D

class FireModule(tf.keras.Model):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        
        self.squeeze = Conv2D(squeeze_channels, kernel_size=1, input_shape=(in_channels,))
        self.squeeze_activation = ReLU()
        
        self.expand1x1 = Conv2D(expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = ReLU()
        
        self.expand3x3 = Conv2D(expand3x3_channels, kernel_size=3, padding='same')
        self.expand3x3_activation = ReLU()
    
    def call(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return Concatenate(axis=1)([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ])

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.features = tf.keras.Sequential([
            Conv2D(96, kernel_size=7, strides=2, input_shape=(None, None, 3)),
            ReLU(),
            MaxPooling2D(pool_size=3, strides=2, padding='same'),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            MaxPooling2D(pool_size=3, strides=2, padding='same'),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            MaxPooling2D(pool_size=3, strides=2, padding='same'),
            FireModule(512, 64, 256, 256),
        ])
        
        self.classifier = tf.keras.Sequential([
            Dropout(rate=0.0),
            Conv2D(num_classes, kernel_size=1),
            ReLU(),
            AdaptiveAveragePooling2D((1, 1))
        ])
    
    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return tf.reshape(x, (-1, num_classes))

# Test code
batch_size = 64
input_channels = 3
height = 512
width = 512
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, input_channels, height, width))]

def get_init_inputs():
    return [num_classes]
