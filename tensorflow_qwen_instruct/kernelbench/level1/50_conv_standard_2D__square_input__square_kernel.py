import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same')

    def call(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 256
num_classes = 1000

def get_inputs():
    return [tf.random.normal((batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
