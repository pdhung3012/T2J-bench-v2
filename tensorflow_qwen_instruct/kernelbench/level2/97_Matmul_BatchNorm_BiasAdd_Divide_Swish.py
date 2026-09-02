import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Add, Div, Activation

class Model(tf.keras.Model):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.bn = BatchNormalization(epsilon=bn_eps, momentum=bn_momentum)
        self.bias = tf.Variable(tf.random.normal(bias_shape))
        self.divide_value = divide_value

    def call(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * tf.math.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
