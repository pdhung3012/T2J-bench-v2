import tensorflow as tf
import tensorflow.keras.layers as nn

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(tf.keras.layers.Layer):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def call(self, x):
        return 0.5 * x * (1.0 + tf.math.tanh(
            tf.math.sqrt(2.0 / tf.constant(math.pi)) * (x + 0.044715 * tf.math.pow(x, 3.0))
        ))

class Model(tf.keras.Model):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    It is possible to use tf.keras.layers.MultiHeadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super(Model, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = tf.keras.layers.Dense(3 * n_embd)
        # output projection
        self.c_proj = tf.keras.layers.Dense(n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", tf.linalg.band_part(tf.ones((max_seqlen, max_seqlen)), -1, 0))
        self.n_head = n_head
        self.n_embd = n_embd

    def call(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(C // self.n_head, axis=-1)
        k = tf.reshape(k, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = tf.reshape(q, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = tf.reshape(v, (B, T, self.n_head, C // self.n_head)).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (1.0 / tf.math.sqrt(k.shape[-1]))
        att = att + self.bias[:,:,:T,:T]
        att = tf.nn.relu(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).contiguous().reshape(B, T, C) # re-assemble all head outputs side by side

        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [tf.random.normal([batch_size, max_seqlen, n_embd])]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
