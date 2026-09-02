import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Scale, Softmax

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication (Gemm), Batch Normalization, scaling, and Softmax.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_shape=(in_features,))
        self.bn = BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)
        self.scale = Scale(scale_shape)
        self.softmax = Softmax(axis=1)

    def call(self, inputs):
        """
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(inputs)
        x = self.bn(x)
        x = self.scale(x)
        x = self.softmax(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
