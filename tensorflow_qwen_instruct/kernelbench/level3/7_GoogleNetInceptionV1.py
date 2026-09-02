import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, concatenate

class InceptionModule(keras.Model):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = Conv2D(out_1x1, kernel_size=1, padding='same')
        
        # 3x3 convolution branch
        self.branch3x3 = keras.Sequential([
            Conv2D(reduce_3x3, kernel_size=1, padding='same'),
            Conv2D(out_3x3, kernel_size=3, padding='same')
        ])
        
        # 5x5 convolution branch
        self.branch5x5 = keras.Sequential([
            Conv2D(reduce_5x5, kernel_size=1, padding='same'),
            Conv2D(out_5x5, kernel_size=5, padding='same')
        ])
        
        # Max pooling branch
        self.branch_pool = keras.Sequential([
            AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            Conv2D(pool_proj, kernel_size=1, padding='same')
        ])
    
    def call(self, inputs):
        """
        :param inputs: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(inputs)
        branch3x3 = self.branch3x3(inputs)
        branch5x5 = self.branch5x5(inputs)
        branch_pool = self.branch_pool(inputs)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return concatenate(outputs, axis=1)

class Model(keras.Model):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        self.conv2 = Conv2D(64, kernel_size=1)
        self.conv3 = Conv2D(192, kernel_size=3, padding='same')
        self.maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = AveragePooling2D(pool_size=(1, 1))
        self.dropout = keras.layers.Dropout(0.0)
        self.fc = Dense(1024)
    
    def call(self, inputs):
        """
        :param inputs: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.maxpool1(tf.nn.relu(self.conv1(inputs)))
        x = tf.nn.relu(self.conv2(x))
        x = self.maxpool2(tf.nn.relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [tf.random.normal([batch_size, input_channels, height, width])]

def get_init_inputs():
    return [num_classes]
