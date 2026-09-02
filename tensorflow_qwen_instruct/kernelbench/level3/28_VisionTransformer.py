import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, LayerNormalization, Conv2D, Flatten, Reshape, RandomDropout, GELU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TransformerEncoderLayer, TransformerEncoder

class Model(tf.keras.Model):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(Model, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = tf.Variable(tf.random.normal([1, num_patches + 1, dim]))
        self.patch_to_embedding = Dense(patch_dim, input_shape=(image_size, image_size, channels))
        self.cls_token = tf.Variable(tf.random.normal([1, 1, dim]))
        self.dropout = RandomDropout(emb_dropout)
        
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = tf.keras.layers.Lambda(lambda x: x[:, 0])
        self.mlp_head = tf.keras.Sequential([
            Dense(mlp_dim, input_shape=(dim,)),
            GELU(),
            RandomDropout(dropout),
            Dense(num_classes)
        ])
    
    def call(self, img):
        """
        Forward pass of the Vision Transformer.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        p = self.patch_size
        
        x = tf.image.extract_patches(
            images=img,
            sizes=[1, p, p, 1],
            strides=[1, p, p, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(tf.shape(img)[0], -1, -1)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [tf.random.normal((2, channels, image_size, image_size))]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
