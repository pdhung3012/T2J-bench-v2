import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dropout, Flatten, Dense

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(Model, self).__init__()
        
        # First convolutional layer
        self.conv1 = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same')
        self.relu1 = ReLU()
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        
        # Second convolutional layer
        self.conv2 = Conv2D(256, kernel_size=(5, 5), padding='same')
        self.relu2 = ReLU()
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        
        # Third convolutional layer
        self.conv3 = Conv2D(384, kernel_size=(3, 3), padding='same')
        self.relu3 = ReLU()
        
        # Fourth convolutional layer
        self.conv4 = Conv2D(384, kernel_size=(3, 3), padding='same')
        self.relu4 = ReLU()
        
        # Fifth convolutional layer
        self.conv5 = Conv2D(256, kernel_size=(3, 3), padding='same')
        self.relu5 = ReLU()
        self.maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))
        
        # Fully connected layers
        self.fc1 = Dense(units=4096)
        self.relu6 = ReLU()
        self.dropout1 = Dropout(rate=0.0)
        
        self.fc2 = Dense(units=4096)
        self.relu7 = ReLU()
        self.dropout2 = Dropout(rate=0.0)
        
        self.fc3 = Dense(units=num_classes)
    
    def call(self, inputs):
        """
        :param inputs: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        x = tf.reshape(x, [-1, 256 * 6 * 6])
        
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

# Test code
batch_size = 1024
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
