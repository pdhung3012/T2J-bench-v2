import tensorflow as tf
import tensorflow.keras.layers as layers
import math

class Model(tf.keras.Model):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self, x):
        return 0.5 * x * (1.0 + tf.math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3.0))))

batch_size = 8192
dim = 8192

def get_inputs():
    return [tf.random.normal([batch_size, dim])]

def get_init_inputs():
    return []
