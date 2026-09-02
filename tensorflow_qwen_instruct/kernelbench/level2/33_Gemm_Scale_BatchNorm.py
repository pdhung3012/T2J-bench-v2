import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

class Model(tf.keras.Model):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_dim=in_features)
        self.scale = tf.Variable(RandomNormal(stddev=1.)(shape=scale_shape), trainable=True)
        self.bn = BatchNormalization(epsilon=eps, momentum=momentum)

    def call(self, inputs):
        x = self.gemm(inputs)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
