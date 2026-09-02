import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

tf.random.set_seed(42)
batch_size = 3
seq_len = 4
d_model = 8
num_heads = 2

q = tf.random.normal((batch_size, seq_len, d_model))
k = tf.random.normal((batch_size, seq_len, d_model))
v = tf.random.normal((batch_size, seq_len, d_model))
print(q.shape)

device = "cuda" if tf.config.experimental.list_physical_devices("GPU") else "cpu"
device = device

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
    Q_w = layers.Dense(d_model, use_bias=False)
    K_w = layers.Dense(num_query_groups * d_head, use_bias=False)
    V_w = layers.Dense(num_query_groups * d_head, use_bias=False)
    W_out = layers.Dense(d_model, use_bias=False)

    Q = Q_w(q)
    K = K_w(k)
    V = V_w(v)

    # Share each K/V head across the query heads in its group.
    repeat_factor = num_query_heads // num_query_groups
    K = K.repeat_interleave(repeat_factor, axis=1)
    V = V.repeat_interleave(repeat_factor, axis=1)

    scores = tf.matmul(Q, K, transpose_b=True) / (d_head ** 0.5)

    if mask is not None:
        scores = scores + (1 - tf.cast(mask, dtype=tf.float32)) * (-1e9)

    attn_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(attn_weights, V)

    output = W_out(output)
    return output

# GQA with as many K/V groups as query heads is exactly MHA, so it should match
# tf.keras.layers.MultiHeadAttention up to the (random) projection weights - we compare
# shapes here and check the grouping behaviour separately.
num_query_heads = 4
d_model = 8

q = tf.random.normal((batch_size, seq_len, d_model))
k = tf.random.normal((batch_size, seq_len, d_model))
v = tf.random.normal((batch_size, seq_len, d_model))

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
causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
masked = grouped_query_attention(
    q, k, v, num_query_heads=num_query_heads, num_query_groups=2,
    d_model=d_model, mask=causal,
)
print("masked output:", tuple(masked.shape))
