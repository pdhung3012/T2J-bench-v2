import tensorflow as tf
from tensorflow.keras.layers import Dense, GroupNormalization, LeakyReLU

class Model(tf.keras.Model):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = Dense(hidden_size, input_shape=(input_size,))
        self.gn = GroupNormalization(groups=num_groups, epsilon=eps)
        self.leaky_relu = LeakyReLU(alpha=negative_slope)

    def call(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
num_groups = 512

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
