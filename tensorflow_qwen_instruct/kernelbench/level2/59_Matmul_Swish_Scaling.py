import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.scaling_factor = scaling_factor

    def call(self, x):
        x = self.matmul(x)
        x = x * tf.nn.sigmoid(x)  # Swish activation
        x = x * self.scaling_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
