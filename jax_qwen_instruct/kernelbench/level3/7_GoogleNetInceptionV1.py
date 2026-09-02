import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, Relu, Adagrad, Initializer

def inception_module(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
    net = [
        Conv(in_channels, out_1x1, (1, 1), initializer=Initializer(jnp.zeros)),
        Conv(out_1x1, out_3x3, (3, 3), padding=(1, 1), initializer=Initializer(jnp.zeros)),
        Conv(out_1x1, out_5x5, (5, 5), padding=(2, 2), initializer=Initializer(jnp.zeros)),
        MaxPool(out_1x1, (3, 3), strides=(1, 1), padding=(1, 1), initializer=Initializer(jnp.zeros)),
        Conv(out_1x1, pool_proj, (1, 1), initializer=Initializer(jnp.zeros))
    ]
    return jax.eval_shape(Conv, in_channels, out_1x1, (1, 1)), jax.eval_shape(jax.vmap(lambda m: m(net)), jnp.arange(out_1x1))

class InceptionModule:
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        self.branch1x1, self.branch1x1_params = inception_module(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)
        self.branch3x3, _ = inception_module(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)
        self.branch5x5, _ = inception_module(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)
        self.branch_pool, _ = inception_module(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return jnp.concatenate(outputs, axis=1)

class Model:
    def __init__(self, num_classes=1000):
        self.conv1 = Conv(3, 64, (7, 7), strides=(2, 2), padding=(3, 3), initializer=Initializer(jnp.ones))
        self.maxpool1 = MaxPool((3, 3), strides=(2, 2), padding=(1, 1), initializer=Initializer(jnp.zeros))
        self.conv2 = Conv(64, 64, (1, 1), initializer=Initializer(jnp.zeros))
        self.conv3 = Conv(64, 192, (3, 3), padding=(1, 1), initializer=Initializer(jnp.ones))
        self.maxpool2 = MaxPool((3, 3), strides=(2, 2), padding=(1, 1), initializer=Initializer(jnp.zeros))
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool((3, 3), strides=(2, 2), padding=(1, 1), initializer=Initializer(jnp.zeros))
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool((3, 3), strides=(2, 2), padding=(1, 1), initializer=Initializer(jnp.zeros))
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = MaxPool((1, 1), strides=(1, 1), padding=(0, 0), initializer=Initializer(jnp.zeros))
        self.dropout = Adagrad(learning_rate=0.001)
        self.fc = Dense(num_classes, initializer=Initializer(jnp.ones))

    def forward(self, x):
        x = self.maxpool1(Relu(self.conv1(x)))
        x = Relu(self.conv2(x))
        x = self.maxpool2(Relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = jnp.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(0), (batch_size, input_channels, height, width))]

def get_init_inputs():
    return [num_classes]
