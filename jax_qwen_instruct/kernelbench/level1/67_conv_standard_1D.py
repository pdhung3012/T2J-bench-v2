import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv1D, Dense, MaxPool1D, Flatten, BatchNorm, Relu, LogSoftmax

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        net = [
            Conv1D(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias),
            BatchNorm(),
            Relu(),
            MaxPool1D(pool_size=2, strides=2),
            Flatten(),
            Dense(1024, bias=True),
            Relu(),
            Dense(512, bias=True),
            Relu(),
            Dense(128, bias=True),
            Relu(),
            Dense(1, bias=True),
            LogSoftmax()
        ]
        self.net = net
        
    def init_weights_and_state(self, key):
        params = self.init(key, jnp.ones((1, self.in_channels, self.kernel_size)), {})
        return params, {}
    
    def __call__(self, inputs, state):
        return self.apply(state, inputs)

key = jax.random.PRNGKey(0)
model = Model(in_channels, out_channels, kernel_size)
params, state = model.init_weights_and_state(key)
output = model(jnp.ones((1, in_channels, length)), state)(jnp.ones((1, in_channels, length)))
output.shape
