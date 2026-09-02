import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.layers as kl
import math

# Synthetic data
tf.random.set_seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = tf.random.normal((batch_size, seq_len, d_model))
k = tf.random.normal((batch_size, seq_len, d_model))
v = tf.random.normal((batch_size, seq_len, d_model))
print(q.shape)

device = "cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu"
device = "cpu"

class SinusoidalPositionalEmbedding(kl.Layer):
    def __init__(self, max_seq_len: int, d_model: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name='position_embeddings',
            shape=(1, self.max_seq_len, self.d_model),
            initializer=kl.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_seq_len, delta=1)
        position_embeddings = self.position_embeddings[:, :tf.shape(x)[1], :]
        return x + position_embeddings

max_seq_len = 100
d_model = 64

# Generate embeddings for a sequence of length 50
seq_len = 50
positions = tf.range(seq_len, dtype=tf.float32).reshape((1, seq_len))  # Shape: (1, seq_len)
custom_pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)

positional_encoding_custom = custom_pos_emb(tf.convert_to_tensor(positions))

print(positional_encoding_custom.shape)  # (1, 50, 64)
