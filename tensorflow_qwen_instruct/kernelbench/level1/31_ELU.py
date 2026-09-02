import tensorflow as tf
import tensorflow.keras.layers as KL

class Model(tf.keras.Model):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha=1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha
    
    def call(self, inputs):
        """
        Applies ELU activation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of any shape.

        Returns:
            tf.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return tf.keras.activations.elu(inputs, alpha=self.alpha)

batch_size = 4096
dim = 393216

def get_inputs():
    x = tf.random.uniform((batch_size, dim))
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization
