import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, inputs):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            inputs (list): List containing two tensors A and B.

        Returns:
            tf.Tensor: Output tensor C.
        """
        A, B = inputs
        return tf.matmul(A, B)

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = tf.random.normal([N, N])
    A = (A + tf.transpose(A)) / 2  # Ensure symmetry
    B = tf.random.normal([N, N])
    B = (B + tf.transpose(B)) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []
