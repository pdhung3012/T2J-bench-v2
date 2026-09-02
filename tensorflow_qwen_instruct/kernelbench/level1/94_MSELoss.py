import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

class Model(tf.keras.Model):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.dense = Dense(dim, input_shape=input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        return tf.reduce_mean(tf.square(x - inputs))

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = tf.random.uniform(())
    return [tf.random.normal([batch_size] + list(input_shape)) * scale, tf.random.normal([batch_size] + list(input_shape))]

def get_init_inputs():
    return []
