import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation

class Model(tf.keras.Model):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.dense = Dense(out_features, input_shape=(in_features,))
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = tf.nn.gelu(x)
        x = tf.nn.softmax(x, axis=1)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
