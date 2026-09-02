import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU

class Model(tf.keras.Model):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()
        
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            self.add_layer(Dense(layer_size, input_shape=(current_input_size,)))
            self.add_layer(ReLU())
            current_input_size = layer_size
        
        self.add_layer(Dense(output_size))
    
    def call(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.call(x)

# Test code
batch_size = 128
input_size = 16384
layer_sizes = [16384, 16384]
output_size = 8192

def get_inputs():
    return [tf.random.normal((batch_size, input_size))]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
