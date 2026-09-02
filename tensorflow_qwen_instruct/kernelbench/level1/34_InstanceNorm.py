import tensorflow as tf
from tensorflow.keras.layers import InstanceNormalization

class Model(tf.keras.Model):
    """
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = InstanceNormalization(axis=1, epsilon=1e-05, gamma_initializer='ones', beta_initializer='zeros')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            tf.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)

batch_size = 112  # heavier workload
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
