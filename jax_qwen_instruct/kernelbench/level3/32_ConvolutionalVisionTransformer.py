import jax
import jax.numpy as jnp
from jax import random
from jax.nn import init

class Model(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(jnp.zeros((1, 1, embed_dim)))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(start_axis=1)  # (B, embed_dim * num_patches)
        x = self.linear_proj(x)  # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, 1, -1)  # (B, 1, embed_dim)
        x = jnp.concatenate((cls_tokens, x[:, None]), axis=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])  # Use [CLS] token for classification

# === Test config ===
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs(seed):
    key = random.PRNGKey(seed)
    x = random.normal(key, (batch_size, in_channels, image_size, image_size))
    return x

def get_init_inputs(seed):
    key = random.PRNGKey(seed)
    num_classes, embed_dim, num_heads = init.init_params(
        lambda rng, key: Model(num_classes, embed_dim, num_heads),
        {'key': key}
    )
    return num_classes, embed_dim, num_heads
