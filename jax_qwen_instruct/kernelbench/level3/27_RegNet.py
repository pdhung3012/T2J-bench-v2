import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental.stax import Conv, BatchNorm, Relu, MaxPool, Dense

def regnet_model(input_channels, stages, block_widths, output_classes):
    def init_params(key):
        key, *rest = jax.random.split(key, 1 + len(block_widths))
        features = [Conv(block_widths[0], (3, 3), padding='SAME'), 
                    BatchNorm(), 
                    Relu()]
        for i in range(1, stages):
            features.append(Conv(block_widths[i], (3, 3), padding='SAME'))
            features.append(BatchNorm())
            features.append(Relu())
            features.append(MaxPool((2, 2), strides=(2, 2)))
        features.append(Dense(output_classes, name='fc'))
        net = nn.Sequential(features)
        return net

    def loss(params, batch):
        logits = net.apply(params, batch['image'])
        labels = batch['label']
        loss_value = F.cross_entropy(logits, labels)
        return loss_value

    return init_params, loss

batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, shape=(batch_size, input_channels, image_height, image_width))
    labels = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=output_classes)
    return {'image': inputs, 'label': labels}

def get_init_inputs():
    return [input_channels, stages, block_widths, output_classes]
