import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import ConvTranspose1D, Dense, Relu, Flatten, BatchNorm, MaxPool1D, ScaleShift

class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = ConvTranspose1D(out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.dense = Dense(128)
        self.relu = Relu()
        self.batchnorm = BatchNorm()
        self.maxpool = MaxPool1D(pool_size=2, strides=2)
        self.flatten = Flatten()
        self.dense_out = Dense(1)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1d_transpose(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense_out(x)
        return x

# Test code
batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
