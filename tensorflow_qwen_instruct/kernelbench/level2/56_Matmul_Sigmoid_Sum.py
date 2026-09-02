import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.dense = Dense(hidden_size, input_shape=(input_size,))
        self.activation = Activation('sigmoid')

    def call(self, inputs):
        """
        Args:
            inputs: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.dense(inputs)
        x = self.activation(x)
        x = tf.reduce_sum(x, axis=1, keepdims=True)
        return x

batch_size = 128
input_size = 32768
hidden_size = 32768

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size]
