import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(Model, self).__init__()
        
        # VGG16 architecture: 5 blocks of convolutional layers followed by max pooling
        self.features = tf.keras.Sequential([
            # Block 1
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            # Block 2
            Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            # Block 3
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            # Block 4
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            # Block 5
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        ])
        
        # Fully connected layers
        self.classifier = tf.keras.Sequential([
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(rate=0.0),
            Dense(4096, activation='relu'),
            Dropout(rate=0.0),
            Dense(num_classes)
        ])
    
    def call(self, inputs):
        """
        Forward pass of the VGG16 model.
        
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
