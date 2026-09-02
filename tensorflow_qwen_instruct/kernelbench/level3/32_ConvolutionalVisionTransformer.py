import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TransformerEncoderLayer

class CViT(Model):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=6, mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super(CVIiT, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = Conv2D(in_channels, embed_dim, kernel_size=patch_size, strides=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = Dense(embed_dim * num_patches)

        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.0, batch_first=True))

        self.cls_token = tf.Variable(tf.zeros([1, 1, embed_dim]))
        self.fc_out = Dense(num_classes)

    def call(self, x):
        B = tf.shape(x)[0]
        x = self.conv1(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = Flatten()(x)                   # (B, embed_dim * num_patches)
        x = self.linear_proj(x)            # (B, embed_dim)

        cls_tokens = self.cls_token.expand_dims(1)  # (B, 1, embed_dim)
        x = tf.concat([cls_tokens, x], axis=1)      # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])        # Use [CLS] token for classification

# === Test config ===
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [tf.random.normal([batch_size, in_channels, image_size, image_size])]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
