import math
import torch

# Triton needs a GPU. Guard the import so the notebook still opens, and the
# PyTorch reference below still runs, on a CPU-only machine such as a laptop or
# a free Colab runtime without an accelerator.
CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

print(f"CUDA available: {CUDA_AVAILABLE} | Triton available: {TRITON_AVAILABLE}")

if TRITON_AVAILABLE:
    @triton.jit
    def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, 
                         O_ptr, L_ptr,
                         stride_qb, stride_qq, stride_qd,
                         stride_kb, stride_kk, stride_kd,
                         stride_vb, stride_vk, stride_vd,
                         stride_ob, stride_ok, stride_od,
                         stride_lb, stride_lq,
                         N_q, N_k,
                         scale,
                         D: tl.constexpr,
                         BLOCK_SIZE_Q: tl.constexpr,
                         BLOCK_SIZE_K: tl.constexpr):
        
        # Program Indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
    
        # Block pointers
        Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                        shape=(N_q, D),
                                        strides=(stride_qq, stride_qd),
                                        offsets=(query_tile_index * BLOCK_SIZE_Q, 0),
                                        block_shape=(BLOCK_SIZE_Q, D),
                                        order=(1,0))
        
        K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                        shape=(D, N_k),
                                        strides=(stride_kd, stride_kk),
                                        offsets=(0, 0),
                                        block_shape=(D, BLOCK_SIZE_K),
                                        order=(0,1)) # Note: K is transposed in the kernel
            
        V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                        shape=(N_k, D),
                                        strides=(stride_vk, stride_vd),
                                        offsets=(0, 0),
                                        block_shape=(BLOCK_SIZE_K, D),
                                        order=(1,0))
        
        O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                        shape=(N_q, D),
                                        strides=(stride_ok, stride_od),
                                        offsets=(query_tile_index * BLOCK_SIZE_Q, 0),
                                        block_shape=(BLOCK_SIZE_Q, D),
                                        order=(1,0))
        
        L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb,
                                        shape=(N_q,),
                                        strides=(stride_lq,),
                                        offsets=(query_tile_index * BLOCK_SIZE_Q,),
                                        block_shape=(BLOCK_SIZE_Q,),
                                        order=(0,))
        
        l = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0  # Initialize l to 1.0
        out = tl.zeros([BLOCK_SIZE_Q, D], dtype=tl.float32)
    
        prev_max = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')  # Initialize s_max to negative infinity
    
        # Load query
        q = tl.load(Q_block_ptr).to(tl.float32)
    
        for i in range(0, N_k, BLOCK_SIZE_K):
    
            # Load keys and values
            k = tl.load(K_block_ptr).to(tl.float32)
            v = tl.load(V_block_ptr).to(tl.float32)
    
            # Compute the attention scores
            s = tl.dot(q, k) * scale
            curr_max = tl.maximum(prev_max, tl.max(s, axis=1))
            p = tl.math.exp(s - curr_max[:, None])
    
    
            # Compute the output
            alpha = tl.math.exp(prev_max - curr_max)
            out = out * alpha[:, None] + tl.dot(p, v)
    
            # To store the logsumexp for backward pass
            curr_l = tl.sum(p, axis=1)
            l = l * alpha + curr_l
    
            prev_max = curr_max
    
            # Advance block pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_K))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_K, 0))
    
        out = out / l[:, None]  # Normalize the output
        tl.store(O_block_ptr, out.to(O_ptr.dtype.element_ty))
    
        # Store the logsumexp
        log_l = prev_max + tl.log(l)
        tl.store(L_block_ptr, log_l.to(L_ptr.dtype.element_ty))


def flash_attention_pytorch(Q, K, V, block_size_q=16, block_size_k=16):
    """Tiled ("flash") attention in pure PyTorch - runs anywhere.

    This is the algorithm the Triton kernel implements: walk the keys in blocks,
    keeping a running max and a running sum so the softmax never materialises the
    full N_q x N_k score matrix. Getting the rescale by exp(prev_max - new_max)
    right is the whole exercise.

    Args:
        Q: (B, N_q, D)
        K: (B, N_k, D)
        V: (B, N_k, D)

    Returns:
        O: (B, N_q, D) attention output
        L: (B, N_q) log-sum-exp of the scores, as the backward pass needs
    """
    B, N_q, D = Q.shape
    N_k = K.shape[1]
    scale = 1.0 / torch.sqrt(torch.tensor(D, dtype=torch.float32))

    O = torch.zeros(B, N_q, D, dtype=torch.float32, device=Q.device)
    L = torch.zeros(B, N_q, dtype=torch.float32, device=Q.device)

    Qf, Kf, Vf = Q.float(), K.float(), V.float()

    for q0 in range(0, N_q, block_size_q):
        q1 = min(q0 + block_size_q, N_q)
        q_blk = Qf[:, q0:q1, :]

        running_max = torch.full((B, q1 - q0), float("-inf"), device=Q.device)
        running_sum = torch.zeros(B, q1 - q0, device=Q.device)
        acc = torch.zeros(B, q1 - q0, D, device=Q.device)

        for k0 in range(0, N_k, block_size_k):
            k1 = min(k0 + block_size_k, N_k)
            scores = torch.matmul(q_blk, Kf[:, k0:k1, :].transpose(-2, -1)) * scale

            blk_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, blk_max)

            # Rescale what we accumulated so far to the new maximum.
            correction = torch.exp(running_max - new_max)
            correction = torch.nan_to_num(correction, nan=0.0)

            p = torch.exp(scores - new_max.unsqueeze(-1))
            running_sum = running_sum * correction + p.sum(dim=-1)
            acc = acc * correction.unsqueeze(-1) + torch.matmul(p, Vf[:, k0:k1, :])
            running_max = new_max

        O[:, q0:q1, :] = acc / running_sum.unsqueeze(-1)
        L[:, q0:q1] = running_max + torch.log(running_sum)

    return O, L


# Quick check against PyTorch's own attention.
_B, _Nq, _Nk, _D = 2, 40, 48, 16
torch.manual_seed(0)
_Q = torch.randn(_B, _Nq, _D, dtype=torch.float32)
_K = torch.randn(_B, _Nk, _D, dtype=torch.float32)
_V = torch.randn(_B, _Nk, _D, dtype=torch.float32)

_O, _L = flash_attention_pytorch(_Q, _K, _V)
_expected = torch.nn.functional.scaled_dot_product_attention(_Q, _K, _V)
print("matches torch attention:", torch.allclose(_O, _expected, atol=1e-5))


# The Triton path only runs with a GPU; the PyTorch path above already ran.
if TRITON_AVAILABLE and CUDA_AVAILABLE:
    B, N_q, N_k, D = 1, 64, 128, 256
    BLOCK_SIZE_Q = 16
    BLOCK_SIZE_K = 16

    Q = torch.randn((B, N_q, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, N_k, D), dtype=torch.float16, device="cuda")
    V = torch.randn((B, N_k, D), dtype=torch.float16, device="cuda")

    O = torch.empty((B, N_q, D), dtype=torch.float16, device="cuda")
    L = torch.empty((B, N_q), dtype=torch.float32, device="cuda")

    stride_qb, stride_qq, stride_qd = Q.stride()
    stride_kb, stride_kk, stride_kd = K.stride()
    stride_vb, stride_vk, stride_vd = V.stride()
    stride_ob, stride_ok, stride_od = O.stride()
    stride_lb, stride_lq
