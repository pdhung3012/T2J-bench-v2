import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, LayerNormalization

class Model(tf.keras.Model):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, use_bias=bias)

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x * tf.nn.sigmoid(x)  # Swish activation
        x = x / 2.0
        x = tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)  # Clamp between -1 and 1
        x = tf.nn.tanh(x)  # Tanh activation
        x = tf.clip_by_value(x, clip_value_min=-1.0, clip_value_max=1.0)  # Clamp between -1 and 1
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
