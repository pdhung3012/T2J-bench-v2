import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs L1 normalization.
    """
    def call(self, inputs):
        """
        Applies L1 normalization to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            tf.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return inputs / tf.reduce_mean(tf.abs(inputs), axis=1, keepdims=True)

batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = tf.random.uniform((batch_size, dim))
    return [x]

def get_init_inputs():
    return []
