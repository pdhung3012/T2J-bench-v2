import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
     

# Synthetic data
torch.manual_seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = torch.rand(batch_size, seq_len, d_model)
k = torch.rand(batch_size, seq_len, d_model)
v = torch.rand(batch_size, seq_len, d_model)
print(q.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
     

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        """
        Initializes the sinusoidal positional embedding.
        
        Args:
            max_seq_len (int): Maximum sequence length.
            d_model (int): Embedding dimension.
        """
        ...

    def forward(self, x):
        """
        Returns the positional embedding for a given input tensor.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            Tensor: Positional embeddings of shape (batch_size, seq_len, d_model).
        """
        ...
     

# from fairseq.modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

max_seq_len = 100
d_model = 64

# Fairseq's implementation requires the number of embeddings (seq length) and embedding dim
# pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len, padding_idx=None)

# Generate embeddings for a sequence of length 50
seq_len = 50
positions = torch.arange(seq_len).unsqueeze(0)  # Shape: (1, seq_len)
# positional_encoding = pos_emb(positions)  # Shape: (1, seq_len, d_model)

custom_pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)

positional_encoding_custom = custom_pos_emb(positions)

print(positional_encoding_custom.shape)  # (1, 50, 64)
