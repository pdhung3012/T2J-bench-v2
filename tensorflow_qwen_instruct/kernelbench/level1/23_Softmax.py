import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

class Model(tf.keras.Model):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.dense = Dense(393216, activation='softmax')

    def call(self, inputs):
        return self.dense(inputs)

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.normal([batch_size, dim])
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
