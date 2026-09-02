import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling1D, Subtract, Activation

class Model(tf.keras.Model):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = Dense(out_features)
        self.max_dim = max_dim

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = tf.reduce_max(x, axis=self.max_dim, keepdims=True)
        x = x - tf.reduce_mean(x, axis=1, keepdims=True)
        x = Activation('gelu')(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, max_dim]
