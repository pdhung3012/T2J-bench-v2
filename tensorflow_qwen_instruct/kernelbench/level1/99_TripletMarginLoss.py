import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import TriangularLoss

class Model(tf.keras.Model):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = TriangularLoss(margin=margin)

    def call(self, inputs):
        anchor, positive, negative = inputs
        return self.loss_fn(anchor, positive, negative)

batch_size = 32768
input_shape = (8192,)
dim = 1

def get_inputs():
    scale = tf.random.uniform([])
    return [tf.random.normal([batch_size, *input_shape]) * scale,
            tf.random.normal([batch_size, *input_shape]),
            tf.random.normal([batch_size, *input_shape])]

def get_init_inputs():
    return [1.0]  # Default margin
