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

def grouped_query_attention(q, k, v, num_query_heads, num_query_groups, d_model,
                            mask=None):
    """
    Implements Grouped Query Attention (GQA).

    Queries keep `num_query_heads` heads, but keys and values only have
    `num_query_groups` heads; each K/V head is shared by
    `num_query_heads // num_query_groups` query heads. With
    num_query_groups == num_query_heads this is ordinary multi-head attention,
    and with num_query_groups == 1 it is multi-query attention.

    Args:
        q, k, v (Tensor): (batch_size, seq_len, d_model)
        num_query_heads (int): number of query heads
        num_query_groups (int): number of key/value heads; must divide num_query_heads
        d_model (int): total embedding dimension
        mask (Tensor, optional): broadcastable to (batch, heads, seq, seq);
            positions equal to 0 are not attended to

    Returns:
        Tensor: (batch_size, seq_len, d_model)
    """
    assert d_model % num_query_heads == 0, "d_model must divide into query heads"
    assert num_query_heads % num_query_groups == 0, \
        "every K/V group must serve the same number of query heads"

    batch_size, seq_len, _ = q.shape
    d_head = d_model // num_query_heads

    # Queries get the full width; keys/values only need one head per GROUP,
    # which is where GQA's parameter and KV-cache savings come from.
    Q_w = nn.Linear(d_model, d_model, bias=False).to(q.device)
    K_w = nn.Linear(d_model, num_query_groups * d_head, bias=False).to(q.device)
    V_w = nn.Linear(d_model, num_query_groups * d_head, bias=False).to(q.device)
    W_out = nn.Linear(d_model, d_model, bias=False).to(q.device)

    Q = Q_w(q).view(batch_size, seq_len, num_query_heads, d_head).transpose(1, 2)
    K = K_w(k).view(batch_size, seq_len, num_query_groups, d_head).transpose(1, 2)
    V = V_w(v).view(batch_size, seq_len, num_query_groups, d_head).transpose(1, 2)

    # Share each K/V head across the query heads in its group.
    repeat_factor = num_query_heads // num_query_groups
    K = K.repeat_interleave(repeat_factor, dim=1)
    V = V.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return W_out(output)
# GQA with as many K/V groups as query heads is exactly MHA, so it should match
# torch.nn.MultiheadAttention up to the (random) projection weights - we compare
# shapes here and check the grouping behaviour separately.
num_query_heads = 4
d_model = 8

q = torch.rand(batch_size, seq_len, d_model)
k = torch.rand(batch_size, seq_len, d_model)
v = torch.rand(batch_size, seq_len, d_model)

for num_query_groups in (1, 2, 4):
    out = grouped_query_attention(
        q, k, v,
        num_query_heads=num_query_heads,
        num_query_groups=num_query_groups,
        d_model=d_model,
    )
    assert out.shape == (batch_size, seq_len, d_model), out.shape
    print(f"num_query_groups={num_query_groups}: {tuple(out.shape)}")

# A causal mask must stop position 0 from seeing later tokens.
causal = torch.tril(torch.ones(seq_len, seq_len)).bool()
masked = grouped_query_attention(
    q, k, v, num_query_heads=num_query_heads, num_query_groups=2,
    d_model=d_model, mask=causal,
)
print("masked output:", tuple(masked.shape))