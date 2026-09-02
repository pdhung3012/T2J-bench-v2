import tensorflow as tf
from tensorflow.keras.layers import Dense, GroupNormalization, Activation

class Model(tf.keras.Model):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_dim=in_features)
        self.group_norm = GroupNormalization(groups=num_groups)
        self.multiply_weight = tf.Variable(tf.random.normal(multiply_weight_shape))

    def call(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * tf.nn.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * tf.nn.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
