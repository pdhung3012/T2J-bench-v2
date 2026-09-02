import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = Dense(hidden_size, input_shape=(input_size,))
        self.scaling_factor = scaling_factor

    def call(self, x):
        """
        Forward pass of the model.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = self.gemm(x)
        original_x = x
        x = tf.nn.sigmoid(x)
        x = x * self.scaling_factor
        x = x + original_x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
