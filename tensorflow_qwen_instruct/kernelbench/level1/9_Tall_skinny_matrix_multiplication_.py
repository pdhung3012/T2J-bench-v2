import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def call(self, inputs):
        """
        Performs the matrix multiplication.

        Args:
            inputs (list): List containing two tensors, A and B, of shapes (M, N) or (N, M).

        Returns:
            tf.Tensor: Output tensor of shape (M, N) or (N, M)
        """
        A, B = inputs
        return tf.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_inputs():
    A = tf.random.normal([M, N])
    B = tf.random.normal([N, M])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
