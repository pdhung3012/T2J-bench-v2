import tensorflow as tf
from tensorflow.keras.layers import GroupNormalization

class Model(tf.keras.Model):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.gn = GroupNormalization(groups=num_groups, axis=-1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            tf.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = tf.random.normal((batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features
