import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU

class Model(tf.keras.Model):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in TensorFlow.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
        
        # Fully connected layers
        self.fc1 = Dense(units=120)
        self.fc2 = Dense(units=84)
        self.fc3 = Dense(units=num_classes)
    
    def call(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = ReLU(self.conv1(x))
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = ReLU(self.conv2(x))
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        # Flatten the output for the fully connected layers
        x = Flatten()(x)
        
        # First fully connected layer with ReLU activation
        x = ReLU(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = ReLU(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [tf.random.normal((batch_size, 1, 32, 32))]

def get_init_inputs():
    return [num_classes]
