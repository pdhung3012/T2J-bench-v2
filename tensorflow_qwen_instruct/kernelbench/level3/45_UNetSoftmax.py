import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose

# U-Net Implementation
class DoubleConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = tf.keras.Sequential([
            Conv2D(out_channels, kernel_size=3, padding='same'),
            BatchNormalization(),
            Activation('softmax'),
            Conv2D(out_channels, kernel_size=3, padding='same'),
            BatchNormalization(),
            Activation('softmax')
        ])

    def call(self, x):
        return self.double_conv(x)

class Model(tf.keras.Model):
    def __init__(self, in_channels, out_channels, features):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (will be doubled in each layer)
        """
        super(Model, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = Conv2DTranspose(features * 16, kernel_size=2, strides=2, padding='same')
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = Conv2DTranspose(features * 8, kernel_size=2, strides=2, padding='same')
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = Conv2DTranspose(features * 4, kernel_size=2, strides=2, padding='same')
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = Conv2DTranspose(features * 2, kernel_size=2, strides=2, padding='same')
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        enc1 = self.encoder1(x)
        enc2 = self.pool1(enc1)
        enc3 = self.encoder2(enc2)
        enc4 = self.pool2(enc3)
        enc4 = self.encoder4(enc4)

        bottleneck = self.pool4(enc4)
        bottleneck = self.bottleneck(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = Concatenate()([dec4, enc4])
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = Concatenate()([dec3, enc3])
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = Concatenate()([dec2, enc2])
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = Concatenate()([dec1, enc1])
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64
# Test code for UNet
def get_inputs():
    return [tf.random.normal((batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, features]
