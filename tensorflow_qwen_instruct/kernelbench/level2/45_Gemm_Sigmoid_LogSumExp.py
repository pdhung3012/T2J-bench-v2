import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import activations

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid,
    another Gemm, and computes LogSumExp over features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = Dense(hidden_size, input_shape=(input_size,))
        self.linear2 = Dense(output_size)

    def call(self, x):
        x = self.linear1(x)
        x = activations.sigmoid(x)
        x = self.linear2(x)
        x = tf.reduce_logsumexp(x, axis=1)  # compute LogSumExp over features per sample
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [tf.random.normal([batch_size, input_size])]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
