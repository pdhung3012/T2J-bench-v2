import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
     

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

import torch
import torch.nn as nn
import torch.nn.functional as F




def multi_head_attention(q, k, v, num_heads, d_model, mask=None):
    """
    Implements multi-head attention.
    
    Args:
        q (Tensor): Query tensor of shape (batch_size, seq_len, d_model)
        k (Tensor): Key tensor of shape (batch_size, seq_len, d_model)
        v (Tensor): Value tensor of shape (batch_size, seq_len, d_model)
        num_heads (int): Number of attention heads
        d_model (int): Total embedding dimension
        mask (Tensor, optional): Masking tensor for attention
        
    Returns:
        Tensor: Multi-head attention output of shape (batch_size, seq_len, d_model)
    """
    ...
     

# Testing on data & compare
output_custom = multi_head_attention(q, k, v, num_heads, d_model)
print(output_custom)

multihead_attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True)
output, _ = multihead_attn(q, k, v)
print(output)

assert torch.allclose(output_custom, output, atol=1e-08, rtol=1e-05) # Check if they are close enough.

