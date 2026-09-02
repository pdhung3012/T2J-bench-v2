import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def call(self, predictions, targets):
        return tf.keras.losses.kldivergy(tf.math.log(predictions), targets, reduction=tf.keras.losses.Reduction.BATCH_WEIGHTED_MEAN)

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = tf.random.uniform(())
    return [tf.nn.softmax((tf.random.normal([batch_size, *input_shape]) * scale), axis=-1),
            tf.nn.softmax(tf.random.normal([batch_size, *input_shape]), axis=-1)]

def get_init_inputs():
    return []
