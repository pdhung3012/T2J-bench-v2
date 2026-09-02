import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose, MaxPool, Sum

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = ConvTranspose(out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = MaxPool(2)
        self.max_pool2 = MaxPool(3)

    @jax.jit
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = jnp.sum(x, axis=1, keepdims=True)
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [jnp.random.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
