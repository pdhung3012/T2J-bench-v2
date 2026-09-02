import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, inputs):
        """
        Performs the matrix multiplication.

        Args:
            inputs (list): A list containing two tensors. The first is a 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,). The second is a 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            tf.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        A, B = inputs
        return tf.matmul(A[:, tf.newaxis], B)

M = 4096
N = 4096

def get_inputs():
    A = tf.random.normal([N])
    B = tf.random.normal([N, M])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
