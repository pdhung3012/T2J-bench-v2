import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

tfd = tfp.distributions

class Model(tf.keras.Model):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.dense = Dense(num_classes, activation=None)

    def call(self, inputs):
        x, y = inputs
        logits = self.dense(x)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [tf.random.normal((batch_size, *input_shape)), tf.random.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
