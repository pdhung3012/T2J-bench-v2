import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU

class Model(tf.keras.Model):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()
        
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.add(Dense(hidden_size))
            self.add(ReLU())
            current_input_size = hidden_size
        
        self.add(Dense(output_size))
    
    def call(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.call(x)

# Test code
batch_size = 1024
input_size = 8192
hidden_layer_sizes = [1024] * 16  # deep network with wider layers
output_size = 8192

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
