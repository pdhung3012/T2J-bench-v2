import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

class GroupedQueryAttention(nn.Module):
    num_query_heads : int
    num_query_groups : int
    d_model : int

    def setup(self):
        self.Q_w = nn.Dense(self.d_model, use_bias=False)
        self.K_w = nn.Dense(self.num_query_groups * (self.d_model // self.num_query_groups), use_bias=False)
        self.V_w = nn.Dense(self.num_query_groups * (self.d_model // self.num_query_groups), use_bias=False)
        self.W_out = nn.Dense(self.d_model, use_bias=False)

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        d_head = self.d_model // self.num_query_heads

        Q = self.Q_w(q).reshape(batch_size, seq_len, self.num_query_heads, d_head).transpose((0, 2, 1, 3))
        K = self.K_w(k).reshape(batch_size, seq_len, self.num_query_groups, d_head).transpose((0, 2, 1, 3))
        V = self.V_w(v).reshape(batch_size, seq_len, self.num_query_groups, d_head).transpose((0, 2, 1, 3))

        repeat_factor = self.num_query_heads // self.num_query_groups
        K = K.repeat_interleave(repeat_factor, axis=2)
        V = V.repeat_interleave(repeat_factor, axis=2)

        scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / (d_head**0.5)

        if mask is not None:
            scores = scores.at[~mask].set(-jnp.inf)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attn_weights, V)
        output = output.transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, self.d_model)
        return self.W_out(output)

batch_size, seq_len, d_model = 3, 4, 8
num_query_heads = 4
num_query_groups = 2
d_model = 8

key, value = random.split(random.PRNGKey(42), 2)
q = random.normal(key, (batch_size, seq_len, d_model))
k = random.normal(value, (batch_size, seq_len, d_model))
v = random.normal(value, (batch_size, seq_len, d_model))

for num_query_groups in (1, 2, 4):
    out = GroupedQueryAttention(num_query_heads=num_query_heads, num_query_groups=num_query_groups, d_model=d_model)(q, k, v)
    assert out.shape == (batch_size, seq_len, d_model), out.shape
    print(f"num_query_groups={num_query_groups}: {tuple(out.shape)}")

causal = jnp.tril(jnp.ones((seq_len, seq_len))).astype(jnp.bool_)
masked = GroupedQueryAttention(num_query_heads=num_query_heads, num_query_groups=2, d_model=d_model)(q, k, v, mask=causal)
print("masked output:", tuple(masked.shape))
