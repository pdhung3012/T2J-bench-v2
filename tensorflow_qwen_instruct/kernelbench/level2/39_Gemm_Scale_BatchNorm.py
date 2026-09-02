import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_dim=in_features)
        self.scale = tf.Variable(np.random.randn(*scale_shape), dtype=tf.float32)
        self.bn = BatchNormalization(epsilon=eps, momentum=momentum)

    def call(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
