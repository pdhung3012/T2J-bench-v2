import jax
import jax.numpy as jnp
from jax import random

class NewGELU(jax.namedtuple):
    def __new__(cls, x):
        return super().__new__(cls, 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0)))))

class Model(jax.namedtuple):
    def __new__(self, n_embd, n_head, max_seqlen):
        assert n_embd % n_head == 0
        self.c_attn = jax.namedtuple('c_attn', ['q', 'k', 'v'])(*[jax.random.normal(random.key(), (n_embd, 3 * n_embd)) for _ in range(3)])
        self.c_proj = jax.namedtuple('c_proj', ['out'])(jax.random.normal(random.key(), (n_embd, n_embd)))
        self.bias = jnp.tril(jnp.ones((max_seqlen, max_seqlen))).reshape((1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn.q, self.c_attn.k, self.c_attn.v
        k = k.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        q = q.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        v = v.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))

        att = (q @ k.transpose(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = att.at[:, :, :, :T].set(att[:, :, :, :T] * self.bias)
        att = jax.nn.relu(att)

        y = att @ v
        y = y.transpose((0, 2, 1, 3)).reshape((B, T, C))
        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [jax.random.normal(random.key(), (batch_size, max_seqlen, n_embd))]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
