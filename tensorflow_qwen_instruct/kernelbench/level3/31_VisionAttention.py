import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention

class Model(tf.keras.Model):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm = LayerNormalization(embed_dim)

    def call(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = tf.shape(x)
        x = tf.reshape(x, shape=(B, C, H * W))  # (batch_size, embed_dim, seq_len)
        x = tf.transpose(x, perm=[2, 0, 1])  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = tf.transpose(x, perm=[1, 2, 0])
        x = tf.reshape(x, shape=(B, C, H, W))
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [tf.random.normal((batch_size, num_channels, image_height, image_width))]

def get_init_inputs():
    return [embed_dim, num_heads]
