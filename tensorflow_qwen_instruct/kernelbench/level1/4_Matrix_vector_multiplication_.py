import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, inputs):
        A, B = inputs
        return tf.matmul(A, B)

M = 256 * 8 # 2048
K = 131072 * 8 # 1048576

def get_inputs():
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, 1])
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
