import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

class HuberLossModel(Model):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(HuberLossModel, self).__init__()
        self.dense = Dense(dim, activation=None)

    def call(self, inputs):
        predictions, targets = inputs
        return tf.keras.losses.huber(predictions, targets)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = tf.random.uniform([])
    return [tf.random.normal([batch_size, *input_shape]) * scale, tf.random.normal([batch_size, *input_shape])]

def get_init_inputs():
    return []
