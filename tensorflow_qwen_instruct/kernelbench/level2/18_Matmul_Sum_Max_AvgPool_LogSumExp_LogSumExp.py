import tensorflow as tf
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.dense = Dense(out_features, input_shape=(in_features,))
    
    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.dense(x)  # (batch_size, out_features)
        x = tf.reduce_sum(x, axis=1, keepdims=True) # (batch_size, 1)
        x = tf.reduce_max(x, axis=1, keepdims=True)[0] # (batch_size, 1)
        x = tf.reduce_mean(x, axis=1, keepdims=True) # (batch_size, 1)
        x = tf.math.log(tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)) # (batch_size, 1)
        x = tf.math.log(tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features]
