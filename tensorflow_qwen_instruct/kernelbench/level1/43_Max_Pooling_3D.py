import tensorflow as tf
from tensorflow.keras.layers import MaxPooling3D

class Model(tf.keras.Model):
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = MaxPooling3D(pool_size=(kernel_size, kernel_size, kernel_size),
                                    strides=(stride, stride, stride),
                                    padding=padding,
                                    dilation_rate=(dilation, dilation, dilation),
                                    data_format='channels_last',
                                    pool_mode='max',
                                    return_indices=return_indices,
                                    padding_mode='same',
                                    ceil_mode=ceil_mode)

    def call(self, inputs, training=None, mask=None):
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            tf.Tensor: Output tensor with Max Pooling 3D applied.
        """
        return self.maxpool(inputs)

batch_size = 16
channels = 32
dim1 = 128
dim2 = 128
dim3 = 128
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = tf.random.normal((batch_size, channels, dim1, dim2, dim3))
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
