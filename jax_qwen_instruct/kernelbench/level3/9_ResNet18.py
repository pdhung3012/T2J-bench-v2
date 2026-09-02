import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, MaxPool, Dense, AdaptiveAvgPool2d, Flatten

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv(out_channels, (3, 3), stride=stride, padding=(1, 1), use_bias=False)
        self.bn1 = BatchNorm()
        self.relu = Relu()
        self.conv2 = Conv(out_channels, (3, 3), stride=1, padding=(1, 1), use_bias=False)
        self.bn2 = BatchNorm()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = Conv(64, (7, 7), stride=2, padding=(3, 3), use_bias=False)
        self.bn1 = BatchNorm()
        self.relu = Relu()
        self.maxpool = MaxPool((3, 3), strides=(2, 2), paddings=(1, 1))

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Dense(num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = Conv(out_channels * block.expansion, (1, 1), stride=stride, padding=(0, 0), use_bias=False)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = Conv(out_channels * block.expansion, (1, 1), stride=stride, padding=(0, 0), use_bias=False)

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.fc(x)

        return x

# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), input_shape)]

def get_init_inputs():
    return [num_classes]
