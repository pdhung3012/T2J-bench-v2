import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Relu, Dense, Init, BatchNorm

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = Conv(out_channels, (kernel_size, kernel_size), padding='SAME')
        self.bias = Dense(bias_shape)  

    def forward(self, x):
        x = self.conv(x)
        x = Relu(x)
        x = x + self.bias(x)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

get_inputs = lambda: [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

get_init_inputs = lambda: [in_channels, out_channels, kernel_size, bias_shape]
