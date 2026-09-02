import tensorflow as tf
import tensorflow.keras as keras
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

class Rotary(keras.layers.Layer):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (tf.range(0, dim, 2).numpy() / dim))
        self.register_keras_serializable()
        self.inv_freq = tf.constant(inv_freq, dtype=tf.float32)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def call(self, x, seq_dim=1):
        seq_len = tf.shape(x)[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = tf.range(seq_len, dtype=self.inv_freq.dtype)
            freqs = tf.einsum("i,j->ij", t, self.inv_freq)
            emb = tf.concat([freqs, freqs], axis=-1)
            self.cos_cached = tf.cos(emb)
            self.sin_cached = tf.sin(emb)
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=x1.ndim - 1)


@tf.function
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# Apply RoPE to real query/key tensors.
# Rotary(x) returns the (cos, sin) tables for the sequence length of x,
# and apply_rotary_pos_emb rotates q and k with them.
max_seq_len = 100
d_model = 64
seq_len = 50
batch_size = 2

q = tf.random.normal((batch_size, seq_len, d_model))
k = tf.random.normal((batch_size, seq_len, d_model))

custom_pos_emb = Rotary(d_model)
cos, sin = custom_pos_emb(q, seq_dim=1)

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

print(q_rot.shape, k_rot.shape)  # (2, 50, 64) (2, 50, 64)

# A rotation preserves vector norms - a quick sanity check.
print("norms preserved:",
      tf.reduce_all(tf.math.equal(q_rot.norm(axis=-1), q.norm(axis=-1)), dtype=tf.bool))
