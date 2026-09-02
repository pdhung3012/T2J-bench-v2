import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Activation
import math

class NewGELU(Layer):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def call(self, x):
        return 0.5 * x * (1.0 + tf.math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3.0))))

class CausalSelfAttention(tf.keras.layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use tf.keras.layers.MultiHeadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super(CausalSelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = Dense(3 * n_embd)
        self.c_proj = Dense(n_embd)
        self.attn_dropout = Dropout(attn_pdrop)
        self.resid_dropout = Dropout(resid_pdrop)
        self.register_buffer("bias", tf.linalg.band_part(tf.ones((max_seqlen, max_seqlen)), -1, 0))

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        k = self.split_heads(k, B)
        q = self.split_heads(q, B)
        v = self.split_heads(v, B)

        att = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * (1.0 / math.sqrt(k.shape[-1]))
        att = att + self.bias[:T, :T]
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = tf.transpose(y, perm=[0, 2, 1, 3])
        y = tf.reshape(y, (B, T, C))
        y = self.resid_dropout(self.c_proj(y))
        return y

class Model(tf.keras.Model):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super(Model, self).__init__()
        self.ln_1 = LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.models.Sequential([
            Dense(4 * n_embd, activation='relu'),
            Dense(n_embd, activation=None),
            Dropout(resid_pdrop),
            NewGELU()
        ])
        self.mlpf = self.mlp

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

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
