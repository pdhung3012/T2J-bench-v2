import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.dense = Dense(out_features)
        self.divisor = divisor

    def call(self, x):
        x = self.dense(x)
        x = ReLU()(x)
        x = x / self.divisor
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, divisor]
