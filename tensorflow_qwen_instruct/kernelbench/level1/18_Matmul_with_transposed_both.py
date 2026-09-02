import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, inputs):
        """
        Performs matrix multiplication.

        Args:
            inputs: List containing two tensors of shapes (K, M) and (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        A, B = inputs
        return tf.matmul(tf.transpose(A), tf.transpose(B))

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = tf.random.normal([K, M])
    B = tf.random.normal([N, K])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
