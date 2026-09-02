import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, Dense, Sequential

def model(in_channels, out_channels, kernel_size, stride, expand_ratio):
    use_residual = (stride == 1 and in_channels == out_channels)
    hidden_dim = in_channels * expand_ratio
    
    if expand_ratio != 1:
        expand_conv = Sequential([
            Conv(hidden_dim, (1, 1), strides=(1, 1), padding=((0, 0), (0, 0)), name='expand_conv'),
            BatchNorm(name='bn_expand'),
            Relu(name='relu')
        ])
    
    depthwise_conv = Sequential([
        Conv(hidden_dim, (kernel_size, kernel_size), strides=(stride, stride), padding=((kernel_size-1)//2, (kernel_size-1)//2), groups=hidden_dim, name='depthwise_conv'),
        BatchNorm(name='bn_depthwise'),
        Relu(name='relu')
    ])
    
    project_conv = Sequential([
        Conv(out_channels, (1, 1), strides=(1, 1), padding=((0, 0), (0, 0)), name='project_conv'),
        BatchNorm(name='bn_project')
    ])
    
    def forward_fn(x):
        identity = x
        
        if hasattr(expand_conv, 'call'):
            x = expand_conv(x)
        
        x = depthwise_conv(x)
        x = project_conv(x)
        
        if use_residual:
            x += identity
        
        return x
    
    return forward_fn

batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, in_channels, 224, 224))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
