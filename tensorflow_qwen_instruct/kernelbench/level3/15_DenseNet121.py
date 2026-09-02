import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, Dense, Input, concatenate

class DenseBlock(tf.keras.Model):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))

    def _make_layer(self, in_features: int, growth_rate: int):
        return tf.keras.Sequential([
            BatchNormalization(),
            Activation('relu'),
            Conv2D(in_features, kernel_size=3, padding='same', use_bias=False),
            Dropout(0.0)
        ])

    def call(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = concatenate(features, axis=1)
        return x

class TransitionLayer(tf.keras.Model):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = tf.keras.Sequential([
            BatchNormalization(),
            Activation('relu'),
            Conv2D(num_input_features, kernel_size=1, use_bias=False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        ])

    def call(self, x):
        return self.transition(x)

class Model(tf.keras.Model):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()

        self.features = tf.keras.Sequential([
            Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        ])

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = tf.keras.models.Sequential()
        self.transition_layers = tf.keras.models.Sequential()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.add(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.add(transition)
                num_features = num_features // 2

        self.final_bn = BatchNormalization()
        self.classifier = Dense(num_classes)

    def call(self, x):
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers(i)(x)

        x = self.final_bn(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(1, 1))(x)
        x = Flatten()(x)
        x = self.classifier(x)
        return x

# Testing the DenseNet121 model
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [tf.random.normal((batch_size, 3, height, width))]

def get_init_inputs():
    return [32, num_classes]
