import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def call(self, inputs):
        """
        Performs matrix multiplication.

        Args:
            inputs: A list containing two tensors of shapes (M, K) and (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        A, B = inputs
        return tf.matmul(A, B)

M = 16384 * 2
N = 16384 * 2
K = 32 * 2

def get_inputs():
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, N])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
