import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def call(self, inputs):
        """
        Performs matrix multiplication of A and B.

        Args:
            inputs: List containing two tensors, A of shape (M, K) and B of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        A, B = inputs
        return tf.matmul(A, B)

M = 256
N = 256
K = 131072 * 4

def get_inputs():
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, N])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
