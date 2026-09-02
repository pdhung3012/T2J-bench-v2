import tensorflow as tf
from tensorflow.keras.layers import Dense, GroupNormalization
from tensorflow.keras.layers import MinMaxScaler
from tensorflow.keras.layers import LayerNormalization
import numpy as np

class Model(tf.keras.Model):
    """
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_dim=in_features)
        self.group_norm = GroupNormalization(groups=num_groups)
        self.bias = tf.Variable(np.random.randn(*bias_shape).astype(np.float32))

    def call(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = tf.reduce_min(x, axis=1, keepdims=True)
        x = x + self.bias
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
