import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def call(self, inputs):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            inputs (list): List containing two tensors, where the first is of shape (N, M, K) and the second is of shape (K, L).

        Returns:
            tf.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of the first tensor in inputs and the second tensor in inputs along the last dimension of the first tensor.
        """
        A, B = inputs
        return tf.matmul(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = tf.random.normal([N, M, K])
    B = tf.random.normal([K, L])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
