import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization

class Model(tf.keras.Model):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.group_norm = BatchNormalization(groups=num_groups)

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = tf.math.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
