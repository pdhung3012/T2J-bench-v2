import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import softmax

class Model(tf.keras.Model):
    """
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = Dense(out_features, input_shape=(in_features,))
        self.dropout = Dropout(dropout_p)

    def call(self, x):
        """
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = softmax(x, axis=1)
        return x

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [tf.random.normal([batch_size, in_features])]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
