import jax
import jax.numpy as jnp
from jax.experimental import partitioning
import flax.linen as nn
from flax.training.train_state import TrainState

class Model(nn.Module):
    @nn.compact
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv(features=32, kernel_size=3, strides=2, padding='SAME', use_bias=False)
        self.bn1 = nn.BatchNorm()
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(features=16, kernel_size=3, strides=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(features=24, kernel_size=3, strides=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(features=24, kernel_size=3, strides=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(features=40, kernel_size=5, strides=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(features=40, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(features=80, kernel_size=3, strides=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(features=80, kernel_size=3, strides=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(features=112, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(features=112, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(features=192, kernel_size=5, strides=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(features=192, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(features=192, kernel_size=5, strides=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(features=320, kernel_size=3, strides=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv(features=1280, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        self.bn2 = nn.BatchNorm()
        
        # Fully connected layer
        self.fc = nn.Dense(features=num_classes)
    
    def __call__(self, x):
        x = jax.nn.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = jax.nn.relu(self.bn2(self.conv2(x)))
        x = jax.nn.avg_pool(x, ksize=(1, 1), strides=(1, 1), padding='SAME')
        x = jnp.reshape(x, (-1, x.shape[-1].value))
        x = self.fc(x)
        return x

class MBConv(nn.Module):
    @nn.compact
    def __init__(self, features, kernel_size, strides, expand_ratio):
        super(MBConv, self).__init__()
        
        self.use_residual = (strides == 1 and features == features)
        hidden_dim = features * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Conv(features=hidden_dim, kernel_size=1, strides=1, padding='SAME', use_bias=False)
            self.bn_expand = nn.BatchNorm()
            self.relu6 = nn.relu6
            
        self.depthwise_conv = nn.Conv(features=hidden_dim, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=False, groups=hidden_dim)
        self.bn_depthwise = nn.BatchNorm()
        self.relu6 = nn.relu6
        
        self.project_conv = nn.Conv(features=features, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        self.bn_project = nn.BatchNorm()
    
    def __call__(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.relu6(self.bn_expand(self.expand_conv(x)))
        
        x = self.relu6(self.bn_depthwise(self.depthwise_conv(x)))
        x = self.bn_project(self.project_conv(x))
        
        if self.use_residual:
            x += identity
        
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
