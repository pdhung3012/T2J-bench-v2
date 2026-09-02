import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def call(self, inputs):
        """
        Performs batched matrix multiplication.

        Args:
            inputs: A list containing two tensors of shapes (batch_size, m, k) and (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        A, B = inputs
        return tf.matmul(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = tf.random.normal((batch_size, m, k))
    B = tf.random.normal((batch_size, k, n))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
