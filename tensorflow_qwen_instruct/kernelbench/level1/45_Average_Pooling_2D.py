import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D

class Model(tf.keras.Model):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = AveragePooling2D(pool_size=(kernel_size, kernel_size), strides=stride, padding='same')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tf.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(inputs)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = tf.random.normal((batch_size, channels, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size]
