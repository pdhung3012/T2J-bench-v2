import jax
import jax.numpy as jnp
from jax.experimental import partitioning
from jax.experimental.sparse import COO
from jax.experimental.stax import Dense, Conv, BatchNorm, Relu, MaxPool, ShuffleChannel, Dense as JaxDense

@partitioning.auto_parallel(num_partitions=1)
class ShuffleNetUnit:
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        self.conv1 = Conv(in_channels, out_channels//4, (1, 1), strides=(1, 1), padding='SAME', use_bias=False)
        self.bn1 = BatchNorm()
        
        self.conv2 = Conv(out_channels//4, out_channels//4, (3, 3), strides=(1, 1), padding='SAME', use_bias=False, groups=out_channels//4)
        self.bn2 = BatchNorm()
        
        self.conv3 = Conv(out_channels//4, out_channels, (1, 1), strides=(1, 1), padding='SAME', use_bias=False)
        self.bn3 = BatchNorm()
        
        self.shuffle = ShuffleChannel(groups)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, (1, 1), strides=(1, 1), padding='SAME', use_bias=False),
                BatchNorm()
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = Relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = Relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ChannelShuffle:
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // self.groups
        
        x = x.reshape(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(0, 2, 1, 3, 4).reshape(batch_size, -1, height, width)
        return x

class Model:
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        self.conv1 = Conv(3, stages_out_channels[0], (3, 3), strides=(2, 2), padding='SAME', use_bias=False)
        self.bn1 = BatchNorm()
        self.maxpool = MaxPool((3, 3), strides=(2, 2), padding='SAME')
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = Conv(stages_out_channels[3], 1024, (1, 1), strides=(1, 1), padding='SAME', use_bias=False)
        self.bn5 = BatchNorm()
        
        self.fc = JaxDense(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = Relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = Relu(self.bn5(self.conv5(x)))
        x = MaxPool((1, 1), strides=(1, 1), padding='SAME')(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [jnp.random.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]
