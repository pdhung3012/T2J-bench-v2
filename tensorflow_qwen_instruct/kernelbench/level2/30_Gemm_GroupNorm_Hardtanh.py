import tensorflow as tf
from tensorflow.keras.layers import Dense, GroupNormalization, HardSigmoid

class Model(tf.keras.Model):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardSigmoid.
    """
    def __init__(self, in_features, out_features, num_groups, hard_sigmoid_min, hard_sigmoid_max):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_shape=(in_features,))
        self.group_norm = GroupNormalization(groups=num_groups)
        self.hard_sigmoid = HardSigmoid(min_value=hard_sigmoid_min, max_value=hard_sigmoid_max)

    def call(self, inputs):
        """
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(inputs)
        x = self.group_norm(x)
        x = self.hard_sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hard_sigmoid_min = -2.0
hard_sigmoid_max = 2.0

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, num_groups, hard_sigmoid_min, hard_sigmoid_max]
