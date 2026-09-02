import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, use_bias=False, input_shape=(in_features,))
        self.bias = tf.Variable(tf.random.normal(bias_shape))

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor with shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = ReLU()(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
