import tensorflow as tf
from tensorflow.keras.layers import Layer

class Model(Layer):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            tf.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = tf.sqrt(tf.reduce_mean(x ** 2, axis=1, keepdims=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
