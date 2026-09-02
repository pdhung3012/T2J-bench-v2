import jax
import jax.numpy as jnp
from jax import jit, vmap

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.attn = nn.MultiHeadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    @jit
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose((2, 0, 1))  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.transpose((1, 2, 0)).reshape(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

get_inputs = jit(vmap(lambda: jnp.random.rand(batch_size, num_channels, image_height, image_width)))

get_init_inputs = lambda: [embed_dim, num_heads]
