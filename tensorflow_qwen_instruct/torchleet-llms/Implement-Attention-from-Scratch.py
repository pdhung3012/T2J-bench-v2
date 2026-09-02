import tensorflow as tf
import tensorflow.keras.backend as K

tf.random.set_seed(42)

batch_size = 1
seq_len = 3
dim = 3

q = tf.random.normal((batch_size, seq_len, dim))
k = tf.random.normal((batch_size, seq_len, dim))
v = tf.random.normal((batch_size, seq_len, dim))

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
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask  # Using broadcasting for masking
    
    # Apply softmax to get attention weights along the last dimension
    attention_weights = K.softmax(scores, axis=-1)  # axis=-1 ensures softmax is applied across the last axis
    
    # Compute output by weighting V with the attention weights
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

# Testing on data & compare
output_custom, _ = scaled_dot_product_attention(q, k, v)
print(output_custom.numpy())
output = F.scaled_dot_product_attention(q, k, v)
print(output.numpy())

assert tf.reduce_allclose(output_custom, output, atol=1e-08, rtol=1e-05) # Check if they are close enough.
