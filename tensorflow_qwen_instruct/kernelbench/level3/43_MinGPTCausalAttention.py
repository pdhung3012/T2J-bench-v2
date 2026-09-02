import tensorflow as tf
import tensorflow.keras.layers as nn

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(tf.keras.Model):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use tf.keras.layers.MultiHeadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = tf.keras.layers.Dense(3 * n_embd)
        # output projection
        self.c_proj = tf.keras.layers.Dense(n_embd)
        # regularization
        self.attn_dropout = tf.keras.layers.Dropout(attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", tf.linalg.band_part(tf.ones((max_seqlen, max_seqlen)), -1, 0))
        self.n_head = n_head
        self.n_embd = n_embd

    def call(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v  = tf.split(qkv, 3, axis=-1)
        k = tf.reshape(k, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = tf.reshape(q, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = tf.reshape(v, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (1.0 / tf.math.sqrt(k.shape[-1]))
        att = att + self.bias[:,:,:T,:T]
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [tf.random.normal([batch_size, seq_len, n_embd])]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
