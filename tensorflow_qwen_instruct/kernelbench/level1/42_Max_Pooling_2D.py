import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D

class Model(tf.keras.Model):
    """
    Simple model that performs Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(Model, self).__init__()
        self.maxpool = MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=(stride, stride), padding='same', dilation_rate=(dilation, dilation))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tf.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return self.maxpool(inputs)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = tf.random.normal((batch_size, channels, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
