import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Activation, Mish

class Model(tf.keras.Model):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, use_bias=False)
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.hardtanh = tf.keras.layers.Hardtanh()
        self.mish = Mish()
        self.groupnorm = LayerNormalization(gamma_initializer='ones', beta_initializer='zeros')

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
