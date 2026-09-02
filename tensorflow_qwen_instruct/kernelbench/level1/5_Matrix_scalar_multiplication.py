import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def call(self, inputs):
        """
        Performs matrix-scalar multiplication.

        Args:
            inputs: List containing two elements:
                - A: Input matrix of shape (M, N)
                - s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        A, s = inputs
        return A * s

M = 16384 * 4
N = 4096 * 4

def get_inputs():
    A = tf.random.uniform((M, N))
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed
