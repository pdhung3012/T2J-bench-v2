import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

torch.manual_seed(42)

batch_size = 1
seq_len = 3
dim = 3

q = torch.randn(batch_size, seq_len, dim)
k = torch.randn(batch_size, seq_len, dim)
v = torch.randn(batch_size, seq_len, dim)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute the scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output tensor of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights tensor of shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]  # Get the last dimension size (key dimension)
    
    # Compute the dot product of Q and K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights along the last dimension
    attention_weights = F.softmax(scores, dim=-1)  # dim=-1 ensures softmax is applied across the last axis
    
    # Compute output by weighting V with the attention weights
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

# Testing on data & compare
output_custom, _ = scaled_dot_product_attention(q, k, v)
print(output_custom)
output = F.scaled_dot_product_attention(q, k, v)
print(output)

assert torch.allclose(output_custom, output, atol=1e-08, rtol=1e-05) # Check if they are close enough.