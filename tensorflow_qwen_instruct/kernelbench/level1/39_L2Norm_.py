import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.
        """
        super(Model, self).__init__()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            tf.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return x / tf.norm(x, axis=1, keepdims=True)

batch_size = 32768
# choose dim so total <2^31
dim = 65535

def get_inputs():
    x = tf.random.normal([batch_size, dim])
    return [x]

def get_init_inputs():
    return []
