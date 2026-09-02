import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, AveragePooling2D

class Model(tf.keras.Model):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(Model, self).__init__()
        self.transition = tf.keras.Sequential([
            BatchNormalization(trainable=True),
            ReLU(),
            Conv2D(filters=num_output_features, kernel_size=(1, 1), padding='same', use_bias=False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        ])

    def call(self, inputs):
        """
        :param inputs: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(inputs)

batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [tf.random.normal((batch_size, num_input_features, height, width))]

def get_init_inputs():
    return [num_input_features, num_output_features]
