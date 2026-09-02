import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (tf.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (tf.Tensor): Input matrix of shape (l, k)

    Returns:
        tf.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def call(self, inputs):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            inputs (list): List containing two tensors [A, B]
                where A is of shape (b, i, j, l) and B is of shape (l, k).

        Returns:
            tf.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        A, B = inputs
        return tf.einsum("bijl,lk->bijk", A, B)

# Test code
b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = tf.random.normal((b, i, j, l))
    B = tf.random.normal((l, k))
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
