import tensorflow as tf
from tensorflow.keras.layers import Dense, InstanceNormalization

class Model(tf.keras.Model):
    """
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.bmm = Dense(out_features, use_bias=False)
        self.instance_norm = InstanceNormalization(axis=-1, epsilon=eps, momentum=momentum)

    def call(self, x, y):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
            y (tf.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.bmm(x)
        x = self.instance_norm(x[:, None, None])
        x = x + y
        x = x * y
        return x

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_inputs():
    return [tf.random.normal([batch_size, in_features]), tf.random.normal([batch_size, out_features])]

def get_init_inputs():
    return [in_features, out_features]
