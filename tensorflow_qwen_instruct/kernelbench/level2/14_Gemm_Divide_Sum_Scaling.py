import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = tf.Variable(tf.random.normal([hidden_size, input_size]))
        self.scaling_factor = scaling_factor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = tf.matmul(x, tf.transpose(self.weight))  # Gemm
        x = x / 2  # Divide
        x = tf.reduce_sum(x, axis=1, keepdims=True) # Sum
        x = x * self.scaling_factor  # Scaling
        return x


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
