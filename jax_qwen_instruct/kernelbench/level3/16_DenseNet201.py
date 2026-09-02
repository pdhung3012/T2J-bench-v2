import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, MaxPool, Dense, BatchNorm, Relu, AvgPool, Flatten, Dense as JaxDense

def DenseBlock(num_layers: int, num_input_features: int, growth_rate: int):
    layers = []
    for i in range(num_layers):
        layers.append(_make_layer(num_input_features + i * growth_rate, growth_rate))
    return nn.Sequential(layers)

def _make_layer(in_features: int, growth_rate: int):
    return nn.Sequential([
        BatchNorm(),
        Relu(),
        Conv(in_features, growth_rate, kernel_size=3, padding=1, use_bias=False),
        nn.Dropout(0.0)
    ])

def TransitionLayer(num_input_features: int, num_output_features: int):
    return nn.Sequential([
        BatchNorm(),
        Relu(),
        Conv(num_input_features, num_output_features, kernel_size=1, use_bias=False),
        AvgPool(kernel_size=2, strides=2)
    ])

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            Conv(3, 64, kernel_size=7, strides=2, padding=3, use_bias=False),
            BatchNorm(),
            Relu(),
            MaxPool(kernel_size=3, strides=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = BatchNorm()
        self.classifier = JaxDense(num_features, num_classes)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = jnp.relu(x, inplace=True)
        x = jnp.average(jnp.square(x), axis=(1, 2))
        x = self.classifier(x)
        return x

def get_inputs():
    return [jax.random.normal(next_rng_keys(), (10, 3, 224, 224))]

def get_init_inputs():
    return [32, 10]
