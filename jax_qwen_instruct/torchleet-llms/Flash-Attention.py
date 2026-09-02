import math
import jax
from jax import numpy as jnp
from jax import jit

@jit
def flash_fwd_kernel(Q, K, V, 
                     O, L,
                     stride_qb, stride_qq, stride_qd,
                     stride_kb, stride_kk, stride_kd,
                     stride_vb, stride_vk, stride_vd,
                     stride_ob, stride_ok, stride_od,
                     stride_lb, stride_lq,
                     N_q, N_k,
                     scale,
                     D, BLOCK_SIZE_Q, BLOCK_SIZE_K):
    
    query_tile_index = jnp.int32(jnp.mod(jnp.arange(0, N_q, BLOCK_SIZE_Q), BLOCK_SIZE_Q))
    batch_index = jnp.int32(jnp.floor(jnp.arange(0, N_q, BLOCK_SIZE_Q) / BLOCK_SIZE_Q))
    
    Q_block_ptr = jnp.take_along_axis(Q, query_tile_index[:, None] * jnp.ones((N_q, BLOCK_SIZE_Q), dtype=jnp.int32), axis=1)
    K_block_ptr = jnp.take_along_axis(K, jnp.zeros((N_k, BLOCK_SIZE_K), dtype=jnp.int32), axis=1)
    V_block_ptr = jnp.take_along_axis(V, jnp.zeros((N_k, BLOCK_SIZE_K), dtype=jnp.int32), axis=1)
    
    O_block_ptr = jnp.take_along_axis(O, query_tile_index[:, None] * jnp.ones((N_q, BLOCK_SIZE_Q), dtype=jnp.int32), axis=1)
    L_block_ptr = jnp.take_along_axis(L, query_tile_index[:, None] * jnp.ones((N_q,), dtype=jnp.int32), axis=1)
    
    l = jnp.zeros((BLOCK_SIZE_Q,), dtype=jnp.float32) + 1.0  # Initialize l to 1.0
    out = jnp.zeros((BLOCK_SIZE_Q, D), dtype=jnp.float32)
    
    prev_max = jnp.zeros((BLOCK_SIZE_Q,), dtype=jnp.float32) - float('inf')  # Initialize s_max to negative infinity
    
    q = Q_block_ptr.reshape((-1, D)).astype(jnp.float32)
    
    for i in range(0, N_k, BLOCK_SIZE_K):
    
        k = K_block_ptr[i:i+BLOCK_SIZE_K].reshape((-1, D)).astype(jnp.float32)
        v = V_block_ptr[i:i+BLOCK_SIZE_K].reshape((-1, D)).astype(jnp.float32)
    
        s = jnp.einsum('ij, kj->ik', q, k) * scale
        curr_max = jnp.maximum(prev_max, jnp.max(s, axis=1))
        p = jnp.exp(s - curr_max[:, None])
    
    
        alpha = jnp.exp(prev_max - curr_max)
        out = out * alpha[:, None] + jnp.einsum('ik, kj->ij', p, v)
    
        curr_l = jnp.sum(p, axis=1)
        l = l * alpha + curr_l
    
        prev_max = curr_max
    
        K_block_ptr = jnp.roll(K_block_ptr, BLOCK_SIZE_K, axis=1)
        V_block_ptr = jnp.roll(V_block_ptr, BLOCK_SIZE_K, axis=1)
    
    out = out / l[:, None]  # Normalize the output
    tl.store(O_block_ptr, out.astype(O.dtype))
    
    log_l = prev_max + jnp.log(l)
    tl.store(L_block_ptr, log_l.astype(L.dtype))

def flash_attention_jax(Q, K, V, block_size_q=16, block_size_k=16):
    B, N_q, D = Q.shape
    N_k = K.shape[1]
    scale = 1.0 / jnp.sqrt(D)

    O = jnp.zeros((B, N_q, D), dtype=jnp.float32)
    L = jnp.zeros((B, N_q), dtype=jnp.float32)

    Qf, Kf, Vf = Q.astype(jnp.float32), K.astype(jnp.float32), V.astype(jnp.float32)

    for q0 in range(0, N_q, block_size_q):
        q1 = min(q0 + block_size_q, N_q)
        q_blk = Qf[:, q0:q1, :]

        running_max = jnp.full((B, q1 - q0), float("-inf"))
        running_sum = jnp.zeros((B, q1 - q0))
        acc = jnp.zeros((B, q1 - q0, D))

        for k0 in range(0, N_k, block_size_k):
            k1 = min(k0 + block_size_k, N_k)
            scores = jnp.einsum('ij, kj->ik', q_blk, Kf[:, k0:k1, :].transpose(-2, -1)) * scale

            blk_max = jnp.max(scores, axis=-1)
            new_max = jnp.maximum(running_max, blk_max)

            correction = jnp.exp(running_max - new_max)
            correction = jnp.nan_to_num(correction, nan=0.0)

            p = jnp.exp(scores - new_max[:, None])
            running_sum = running_sum * correction + p.sum(axis=-1)
            acc = acc * correction[:, None] + jnp.einsum('ik, kj->ij', p, Vf[:, k0:k1, :])
            running_max = new_max

        O[:, q0:q1, :] = acc / running_sum[:, None]
        L[:, q0:q1] = running_max + jnp.log(running_sum)

    return O, L


_B, _Nq, _Nk, _D = 2, 40, 48, 16
_Q = jnp.random.randn(_B, _Nq, _D).astype(jnp.float32)
_K = jnp.random.randn(_B, _Nk, _D).astype(jnp.float32)
_V = jnp.random.randn(_B, _Nk, _D).astype(jnp.float32)

_O, _L = flash_attention_jax(_Q, _K, _V)
_expected = jax.nn.functional.scaled_dot_product_attention(_Q, _K, _V)
print("matches torch attention:", jnp.allclose(_O, _expected, atol=1e-5))


if jax.lib.xla_extension.available and jax.lib.xla_extension.runtime_has_cuda():
    B, N_q, N_k, D = 1, 64, 128, 256
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 16

    Q = jnp.random.randn(B, N_q, D).astype(jnp.float16)
    K = jnp.random.randn(B, N_k, D).astype(jnp.float16)
    V = jnp.random.randn(B, N_k, D).astype(jnp.float16)

    O = jnp.empty((B, N_q, D), dtype=jnp.float16)
    L = jnp.empty((B, N_q), dtype=jnp.float32)

    stride_qb, stride_qq, stride_qd = Q.strides
    stride_kb, stride_kk, stride_kd = K.strides
    stride_vb, stride_vk, stride_vd = V.strides
    stride_ob, stride_ok, stride_od = O.strides
    stride_lb, stride_lq = L.strides

    grid = (jax.lax.pmap_dim_size(0) // BLOCK_SIZE_Q, B)

    flash_fwd_kernel[grid](
        Q, K, V, O, L,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_ok, stride_od,
        stride_lb, stride_lq,
        N_q, N_k,
        scale=1.0 / jnp.sqrt(D),
        D=D,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    scale = 1.0 / jnp.sqrt(D)
    scores = jnp.einsum('ij, kj->ik', Q, K.transpose(-2, -1)) * scale
    O_ref = jnp.einsum('ij, ijk->ik', jnp.softmax(scores, axis=-1), V)
    L_ref = jnp.logsumexp(scores, axis=-1)

    print("O matches:", jnp.allclose(O, O_ref, atol=1e-1, rtol=1e-2))
    print("L matches:", jnp.allclose(L, L_ref, atol=1e-1, rtol=1e-2))
else:
    print("Skipping the Triton kernel test - it needs a CUDA GPU with Triton installed.")
