import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", jnp.tril(jnp.ones((max_seqlen, max_seqlen))))
        self.n_head = n_head
        self.n_embd = n_embd

    def _split_heads(self, x, n_heads):
        new_shape = x.shape[:-1] + (n_heads, x.shape[-1] // n_heads)
        return x.view(new_shape).transpose(1, 2)

    def _merge_heads(self, x):
        new_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.transpose(1, 2).reshape(new_shape)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, C // self.n_head).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make masks
        att = (q @ k.transpose(-2, -1)) * (C // self.n_head) ** -0.5
        att = att + self.bias[:T, :T]
        att = jnp.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = self._merge_heads(y)
        y = self.resid_dropout(self.c_proj(y))
        return y

def get_inputs(seed=jax.random.PRNGKey(0)):
    key, dropout_key = random.split(seed)
    return [random.normal(key, (128, 512, 768))]

def get_init_inputs():
    return [768, 8, 0.0, 0.0, 1024]
