import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Add, Sigmoid, Tanh, Activation, Gelu, HardSigmoid, MinMaxScaler

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.add_value = tf.Variable(tf.random.normal(add_value_shape))

    def call(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = Sigmoid()(x) * x  # Swish
        x = Tanh()(x)
        x = Gelu()(x)  # GELU
        x = MinMaxScaler(feature_range=(-1, 1))(x)  # Hardtanh
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [tf.random.normal((batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
