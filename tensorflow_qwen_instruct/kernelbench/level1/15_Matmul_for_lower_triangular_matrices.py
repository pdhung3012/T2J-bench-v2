import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def call(self, inputs):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            inputs (list): A list containing two tensors, A and B, both of shape (M, M).

        Returns:
            tf.Tensor: The result of matrix multiplication C of shape (M, M).
        """
        A, B = inputs
        return tf.linalg.matmul(tf.linalg.band_part(A, -1, 0), tf.linalg.band_part(B, -1, 0))

M = 4096

def get_inputs():
    A = tf.random.normal([M, M])
    B = tf.random.normal([M, M])
    A = tf.linalg.band_part(A, -1, 0)
    B = tf.linalg.band_part(B, -1, 0)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
