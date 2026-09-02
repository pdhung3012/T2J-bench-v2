import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Dropout, Dense, Flatten, Input, Layer, concatenate

class Model(tf.keras.Model):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(Model, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = layers

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNormalization, ReLU, Conv2D, and Dropout.
        """
        return [
            BatchNormalization(),
            Activation('relu'),
            Conv2D(growth_rate, kernel_size=3, padding='same', use_bias=False),
            Dropout(0.0)
        ]
    
    def call(self, inputs):
        """
        :param inputs: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [inputs]
        for layer in self.layers:
            new_feature = layer[1](inputs)  # Apply activation
            features.append(new_feature)
            inputs = concatenate(features, axis=1)  # Concatenate along channel axis
        return inputs
    
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [tf.random.normal((batch_size, num_input_features, height, width))]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
