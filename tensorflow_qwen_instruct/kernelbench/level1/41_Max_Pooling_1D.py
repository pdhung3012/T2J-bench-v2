import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D

class Model(tf.keras.Model):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = MaxPooling1D(pool_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, data_format="channels_last", padding="valid" if padding == 0 else "same", return_indices=return_indices)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            tf.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return self.maxpool(inputs)

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            

return_indices = False

def get_inputs():
    x = tf.random.normal((batch_size, features, sequence_length))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
