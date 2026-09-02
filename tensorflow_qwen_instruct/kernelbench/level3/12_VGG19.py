import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG19 model.

        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(Model, self).__init__()
        
        # VGG19 architecture: 16 Conv layers + 5 MaxPool layers + 3 Fully Connected layers
        self.features = tf.keras.Sequential([
            # Block 1
            Conv2D(64, kernel_size=3, padding='same', input_shape=(224, 224, 3)),
            tf.keras.layers.Activation('relu'),
            Conv2D(64, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=2, strides=2),
            
            # Block 2
            Conv2D(128, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(128, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=2, strides=2),
            
            # Block 3
            Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(256, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=2, strides=2),
            
            # Block 4
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=2, strides=2),
            
            # Block 5
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            Conv2D(512, kernel_size=3, padding='same'),
            tf.keras.layers.Activation('relu'),
            MaxPooling2D(pool_size=2, strides=2)
        ])
        
        self.classifier = tf.keras.Sequential([
            Flatten(),
            Dense(4096),
            tf.keras.layers.Activation('relu'),
            Dropout(rate=0.0),
            Dense(4096),
            tf.keras.layers.Activation('relu'),
            Dropout(rate=0.0),
            Dense(num_classes)
        ])
    
    def call(self, inputs):
        """
        Forward pass of the VGG19 model.

        :param inputs: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(inputs)
        x = Flatten()(x)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
