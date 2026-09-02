import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Scale, Dense, Relu, Flatten, MaxPool, Serial

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.model = Serial(
            Conv(out_channels, (kernel_size, kernel_size), padding='SAME'),
            BatchNorm(),
            Scale(scaling_factor),
            Relu(),
            MaxPool((2, 2)),
            Flatten()
        )

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
