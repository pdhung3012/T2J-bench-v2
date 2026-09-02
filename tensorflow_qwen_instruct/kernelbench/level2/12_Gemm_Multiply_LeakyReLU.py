import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.multiplier = multiplier
        self.leaky_relu = LeakyReLU(negative_slope)

    def call(self, x):
        x = self.gemm(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
