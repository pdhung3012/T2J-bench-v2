import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import RandomUniform

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = Dense(out_features, input_dim=in_features, kernel_initializer=RandomUniform(-1, 1))
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def call(self, x):
        x = self.linear(x)
        x = x - self.subtract_value
        x = x * self.multiply_value
        x = tf.nn.relu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [tf.random.uniform((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
