import jax
import jax.numpy as jnp
from jax import random

class NewGELU(jax.namedtuple):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def __call__(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))

class CausalSelfAttention(jax.namedtuple):
    def __new__(cls, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        self = super(CausalSelfAttention, cls).__new__(cls, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        assert n_embd % n_head == 0
        self.c_attn = jax.namedtuple('c_attn', ['q', 'k', 'v'])(*[jax.random.normal(random.PRNGKey(0), (n_embd,)) for _ in range(3)])
        self.c_attn.q = self.c_attn.q.view(self.n_embd, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        self.c_attn.k = self.c_attn.k.view(self.n_embd, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        self.c_attn.v = self.c_attn.v.view(self.n_embd, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        self.c_proj = jax.namedtuple('c_proj', ['y'])(jax.random.normal(random.PRNGKey(0), (self.n_embd,)))
        self.attn_dropout = jax.namedtuple('attn_dropout', ['att'])(
            jax.nn.Dropout(self.attn_pdrop, deterministic=False)
        )
        self.resid_dropout = jax.namedtuple('resid_dropout', ['y'])(
            jax.nn.Dropout(self.resid_pdrop, deterministic=False)
        )
        self.bias = jnp.tril(jnp.ones((max_seqlen, max_seqlen))).reshape((1, 1, max_seqlen, max_seqlen))
        self.n_head = self.n_head
        self.n_embd = self.n_embd
        return self
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.c_attn.q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.c_attn.k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.c_attn.v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        att = att.atleast_3d().atleast_4d().mask_fill(self.bias[:,:,:T,:T], float('-inf'))
        att = jax.nn.softmax(att, axis=-1)
        att = self.attn_dropout.att(att)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout.y(self.c_proj.y)
        return y
    
class Model(jax.namedtuple):
    def __new__(cls, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        self = super(Model, cls).__new__(cls, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_1 = jax.namedtuple('ln_1', ['x'])(jax.nn.LayerNorm(self.n_embd))
        self.attn = CausalSelfAttention(self.n_embd, self.n_head, self.attn_pdrop, self.resid_pdrop, self.max_seqlen)
        self.ln_2 = jax.namedtuple('ln_2', ['x'])(jax.nn.LayerNorm(self.n_embd))
        self.mlp = jax.namedtuple('mlp', ['c_fc', 'c_proj', 'act', 'dropout'])(*[jax.random.normal(random.PRNGKey(0), (self.n_embd,)) for _ in range(4)])
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
        return self
    
    def forward(self, x):
        x = x + self.attn(self.ln_1.x(x))
        x = x + self.mlpf(self.ln_2.x(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [jax.random.normal(random.PRNGKey(0), (batch_size, seq_len, n_embd))]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
