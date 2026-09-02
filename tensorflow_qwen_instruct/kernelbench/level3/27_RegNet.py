import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

class Model(tf.keras.Model):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(Model, self).__init__()
        
        self.stages = stages
        self.block_widths = block_widths
        
        current_channels = input_channels
        
        # Construct the stages with their respective blocks
        for i in range(stages):
            self.add(tf.keras.Sequential([
                Conv2D(self.block_widths[i], kernel_size=3, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                Conv2D(self.block_widths[i], kernel_size=3, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2), strides=2)
            ]))
            current_channels = self.block_widths[i]
        
        self.feature_extractor = tf.keras.Sequential([
            Conv2D(self.block_widths[-1], kernel_size=1, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ])
        
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = Dense(output_classes)
    
    def call(self, inputs):
        """
        Forward pass through the RegNet model.
        :param inputs: tf.Tensor of shape (batch_size, input_channels, height, width)
        :return: tf.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(inputs)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [tf.random.normal((batch_size, input_channels, image_height, image_width))]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]
