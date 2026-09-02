import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.linear = Dense(out_features, input_shape=(in_features,))
        self.constant = tf.Variable(constant)

    def call(self, x):
        x = self.linear(x)
        x = tf.minimum(x, self.constant)
        x = x - self.constant
        return x

batch_size = 128
in_features = 16384
out_features = 16384
constant = 2.0

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, constant]
