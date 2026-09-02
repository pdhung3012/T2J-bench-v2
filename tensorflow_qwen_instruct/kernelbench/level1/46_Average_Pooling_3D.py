import tensorflow as tf
from tensorflow.keras.layers import AveragePooling3D

class Model(tf.keras.Model):
    """
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = AveragePooling3D(pool_size=(kernel_size, kernel_size, kernel_size),
                                         strides=stride if stride else kernel_size,
                                         padding='valid' if padding == 0 else 'same')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            tf.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool(inputs)

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = tf.random.normal((batch_size, channels, depth, height, width))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
