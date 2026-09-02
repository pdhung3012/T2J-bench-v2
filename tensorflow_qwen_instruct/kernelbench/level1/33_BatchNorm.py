import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

class Model(tf.keras.Model):
    """
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = BatchNormalization(num_features=num_features)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            tf.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(inputs)

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
