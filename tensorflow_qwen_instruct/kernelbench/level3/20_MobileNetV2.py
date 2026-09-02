import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, AveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

class Model(tf.keras.Model):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in TensorFlow.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(Model, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = _make_divisible(inp * expand_ratio, 8)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(Conv2D(hidden_dim, 1, strides=1, padding='same', kernel_initializer=he_normal(),
                                     use_bias=False))
                layers.append(BatchNormalization())
                layers.append(ReLU(6, activation_mode='linear'))

            layers.extend([
                # Depthwise convolution
                DepthwiseConv2D(3, strides=stride, padding='same', depthwise_initializer=he_normal(),
                                use_bias=False),
                BatchNormalization(),
                ReLU(6, activation_mode='linear'),
                # Pointwise linear convolution
                Conv2D(oup, 1, strides=1, padding='same', kernel_initializer=he_normal(),
                       use_bias=False),
                BatchNormalization()
            ])

            if use_res_connect:
                return layers, True
            else:
                return layers, False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [Conv2D(3, input_channel, 3, strides=2, padding='same', kernel_initializer=he_normal(),
                           use_bias=False),
                    BatchNormalization(),
                    ReLU(6, activation_mode='linear')]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.extend(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        features.append(AveragePooling2D(pool_size=1, strides=1, padding='valid'))
        features.append(Conv2D(last_channel, 1, strides=1, padding='same', kernel_initializer=he_normal(),
                                use_bias=False))
        features.append(BatchNormalization())
        features.append(ReLU(6, activation_mode='linear'))

        # Final layer
        features.append(Conv2D(last_channel, 1, strides=1, padding='same', kernel_initializer=he_normal(),
                                use_bias=False))
        features.append(BatchNormalization())

        self.features = tf.keras.Sequential(features)

        # Linear layer
        self.classifier = Dense(num_classes, kernel_initializer=he_normal())

        # Weight initialization
        for m in self.modules():
            if isinstance(m, Conv2D):
                m.kernel_initializer = he_normal()
            elif isinstance(m, BatchNormalization):
                m.gamma_initializer = tf.constant_initializer(1.)
                m.beta_initializer = tf.constant_initializer(0.)
            elif isinstance(m, Dense):
                m.kernel_initializer = he_normal()

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of the MobileNetV2 model.

        :param inputs: The input tensor, shape (batch_size, 3, 224, 224)
        :param training: Whether the model is in training mode.
        :param mask: Not used.
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(inputs)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [tf.random.normal([batch_size, 3, 224, 224])]

def get_init_inputs():
    return [num_classes]
