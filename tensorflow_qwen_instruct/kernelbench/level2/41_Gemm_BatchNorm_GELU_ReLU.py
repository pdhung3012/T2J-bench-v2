import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation

class Model(tf.keras.Model):
    """
    Model that performs a GEMM, BatchNorm, GELU, and ReLU in sequence.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_shape=(in_features,))
        self.batch_norm = BatchNormalization()

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.batch_norm(x)
        x = tf.nn.gelu(x)
        x = tf.nn.relu(x)
        return x

batch_size = 16384
in_features = 4096
out_features = 4096

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
