import jax
import jax.numpy as jnp
import numpy as np
import functools

# Synthetic data
np.random.seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = jnp.random.rand(batch_size, seq_len, d_model)
k = jnp.random.rand(batch_size, seq_len, d_model)
v = jnp.random.rand(batch_size, seq_len, d_model)
print(q.shape)

device = "cuda" if jax.device_count() > 0 else "cpu"
device = device

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.position_embeddings = self._compute_positional_embeddings()

    def _compute_positional_embeddings(self):
        position_enc = np.array([
            [p / np.power(10000, 2 * i / self.d_model) for i in range(self.d_model)]
            if p != 0 else 0
            for p in range(self.max_seq_len)
        ])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        return jnp.array(position_enc)

    def forward(self, x):
        seq_len = x.shape[1]
        position_ids = jnp.arange(seq_len, dtype=jnp.float32)[jnp.newaxis, :]
        position_embeddings = self.position_embeddings[jnp.newaxis, :, :]
        return x + position_embeddings[:, position_ids]

max_seq_len = 100
d_model = 64

# Generate embeddings for a sequence of length 50
seq_len = 50
positions = jnp.arange(seq_len).reshape(1, -1)  # Shape: (1, seq_len)
custom_pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)

positional_encoding_custom = custom_pos_emb(jnp.expand_dims(positions, axis=0))

print(positional_encoding_custom.shape)  # (1, 50, 64)
