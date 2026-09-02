import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import activations

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = Dense(hidden_size, input_shape=(input_size,))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = tf.clip_by_value(x, self.clamp_min, self.clamp_max)
        x = tf.reduce_logsumexp(x, axis=1, keepdims=True)
        x = x * activations.mish(x)  # Mish activation
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
