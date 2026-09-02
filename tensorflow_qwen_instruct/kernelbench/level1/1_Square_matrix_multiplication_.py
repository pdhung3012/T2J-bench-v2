import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def call(self, inputs):
        """
        Performs the matrix multiplication.

        Args:
            inputs (list): List containing two tensors A and B of shape (N, N).

        Returns:
            tf.Tensor: Output tensor C of shape (N, N).
        """
        A, B = inputs
        return tf.matmul(A, B)

N = 2048 * 2

def get_inputs():
    A = tf.random.normal([N, N])
    B = tf.random.normal([N, N])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
