import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

class Model(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    channels: int = 3
    dropout: float = 0.1
    emb_dropout: float = 0.1
    
    def setup(self):
        assert self.image_size % self.patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = self.channels * self.patch_size ** 2
        
        self.patch_size = self.patch_size
        self.pos_embedding = nn.Parameter(jnp.zeros((1, num_patches + 1, self.dim)))
        self.patch_to_embedding = nn.Dense(patch_dim, dtype=jnp.float32)
        self.cls_token = nn.Parameter(jnp.zeros((1, 1, self.dim)), name='cls_token')
        self.dropout = nn.Dropout(self.emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.heads, dim_feedforward=self.mlp_dim, dropout=self.dropout),
            num_layers=self.depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential([
            nn.Dense(self.mlp_dim, dtype=jnp.float32),
            nn.gelu,
            nn.Dropout(self.dropout),
            nn.Dense(self.num_classes, dtype=jnp.float32)
        ])
    
    @nn.compact
    def __call__(self, img):
        p = self.patch_size
        
        x = jax.image.extract_patches(img, sizes=(1, p, p, 1), strides=(1, p, p, 1), padding='VALID').reshape(-1, p*p*self.channels)
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token[jnp.newaxis, :, :]
        x = jnp.concatenate((cls_tokens, x), axis=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs(rng):
    return jax.random.normal(random.PRNGKey(0), (2, channels, image_size, image_size))

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
