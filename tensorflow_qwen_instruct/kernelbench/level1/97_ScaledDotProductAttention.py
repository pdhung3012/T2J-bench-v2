import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

    def call(self, Q, K, V):
        out = MultiHeadAttention(num_heads=32, key_dim=1024)(Q, K, V)
        out = LayerNormalization(epsilon=1e-6)(out + Q)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = tf.random.normal((batch_size, num_heads, sequence_length, embedding_dimension))
    K = tf.random.normal((batch_size, num_heads, sequence_length, embedding_dimension))
    V = tf.random.normal((batch_size, num_heads, sequence_length, embedding_dimension))
    return [Q, K, V]

def get_init_inputs():
    return []
