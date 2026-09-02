import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

class Model(tf.keras.layers.Layer):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def call(self, inputs):
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of arbitrary shape.

        Returns:
            tf.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = tf.linalg.norm(inputs, ord='fro', axis=(-2, -1))
        return inputs / norm

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return []
