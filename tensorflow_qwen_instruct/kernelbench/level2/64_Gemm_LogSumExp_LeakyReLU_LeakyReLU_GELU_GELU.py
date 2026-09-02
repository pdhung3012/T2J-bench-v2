import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Gelu

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = Dense(out_features, use_bias=bias)

    def call(self, x):
        # Gemm
        x = self.linear(x)
        # LogSumExp
        x = tf.reduce_logsumexp(x, axis=1, keepdims=True)
        # LeakyReLU
        x = LeakyReLU(negative_slope=0.01)(x)
        # LeakyReLU
        x = LeakyReLU(negative_slope=0.01)(x)
        # GELU
        x = Gelu()(x)
        # GELU
        x = Gelu()(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
