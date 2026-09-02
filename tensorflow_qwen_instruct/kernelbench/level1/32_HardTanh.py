import tensorflow as tf
import tensorflow.keras.layers as KL

class Model(tf.keras.Model):
    """
    Simple model that performs a HardTanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def call(self, inputs, training=None, mask=None):
        """
        Applies HardTanh activation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of any shape.

        Returns:
            tf.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return tf.keras.activations.hard_sigmoid(inputs)

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.uniform((batch_size, dim))
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
