import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, MaxPooling1D, Activation

class Model(tf.keras.Model):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.avg_pool = GlobalAveragePooling1D(pool_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.avg_pool(x[:, :, tf.newaxis]).squeeze(axis=-1)
        x = tf.nn.gelu(x)
        x = x * self.scale_factor
        x = tf.reduce_max(x, axis=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
