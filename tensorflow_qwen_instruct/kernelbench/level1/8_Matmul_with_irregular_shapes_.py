import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def call(self, inputs):
        """
        Performs matrix multiplication of A and B.

        Args:
            inputs: A list containing two tensors, A and B.
                  A: Input tensor with shape (M, K).
                  B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        A, B = inputs
        return tf.matmul(A, B)

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, N])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
