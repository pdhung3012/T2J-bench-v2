import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization

class Model(tf.keras.Model):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = LayerNormalization(normalized_shape=normalized_shape)

    def call(self, inputs):
        """
        Applies Layer Normalization to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            tf.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(inputs)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
