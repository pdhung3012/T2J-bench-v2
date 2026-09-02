import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def call(self, inputs):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            inputs (list): A list containing two tensors of shape (N, N).

        Returns:
            tf.Tensor: The product of the two tensors in the input list, also an upper triangular matrix of shape (N, N).
        """
        A, B = inputs
        return tf.linalg.matmul(tf.linalg.band_part(A, -1, 0), tf.linalg.band_part(B, -1, 0))

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = tf.linalg.band_part(tf.random.uniform((N, N)), -1, 0)
    B = tf.linalg.band_part(tf.random.uniform((N, N)), -1, 0)
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []
