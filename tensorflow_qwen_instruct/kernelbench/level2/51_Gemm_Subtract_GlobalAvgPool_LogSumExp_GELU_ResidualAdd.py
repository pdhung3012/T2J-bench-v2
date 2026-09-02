import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Add

class Model(tf.keras.Model):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, use_bias=bias)
        self.subtract = tf.Variable(tf.random.normal([out_features]))

    def call(self, x):
        original_x = x.clone().numpy()
        # Gemm
        x = self.gemm(x)

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = tf.reduce_mean(x, axis=1, keepdims=True)

        # LogSumExp
        x = tf.math.log(tf.reduce_sum(tf.exp(x), axis=1, keepdims=True))

        # GELU
        x = tf.nn.gelu(x)

        # ResidualAdd
        x = x + tf.convert_to_tensor(original_x, dtype=tf.float32)

        return x

batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
