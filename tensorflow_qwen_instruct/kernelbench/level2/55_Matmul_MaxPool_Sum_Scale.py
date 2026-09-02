import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPooling1D

class Model(tf.keras.Model):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.max_pool = MaxPooling1D(pool_size=kernel_size, strides=1)
        self.scale_factor = scale_factor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.max_pool(x[:, :, tf.newaxis]).squeeze(axis=-1)
        x = tf.reduce_sum(x, axis=1)
        x = x * self.scale_factor
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
