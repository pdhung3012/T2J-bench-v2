import tensorflow as tf
from tensorflow.keras.layers import Dense, HardSigmoid, GELU

class Model(tf.keras.Model):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = Dense(out_features, input_shape=(in_features,))
        self.scaling_factor = scaling_factor
        self.hardtanh = HardSigmoid(min_value=hardtanh_min, max_value=hardtanh_max)
        self.gelu = GELU()

    def call(self, x):
        x = self.gemm(x)
        x = x * self.scaling_factor
        x = self.hardtanh(x)
        x = self.gelu(x)
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
