import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def call(self, inputs):
        """
        Performs matrix multiplication.

        Args:
            inputs: A list containing two tensors, A of shape (M, K) and B of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        A, B = inputs
        return tf.matmul(A, tf.transpose(B))

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = tf.random.normal([M, K])
    B = tf.random.normal([N, K])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
