import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_length, n_heads))
        self.B = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, n_heads, d_state))
        self.C = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        T = x.shape[-1]
        x_cumsum = jnp.cumsum(x, axis=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = jnp.tril(jnp.ones((T, T)), diagonal=0)
        x_segsum = x_segsum.at[~mask].set(-jnp.inf)
        return x_segsum
    
    @jax.jit
    def forward(self, X, initial_states=None):
        X_blocks = jnp.reshape(X, (self.batch_size, -1, self.n_heads, self.d_head))
        A_blocks = jnp.reshape(self.A, (self.batch_size, -1, self.n_heads))
        B_blocks = jnp.reshape(self.B, (self.batch_size, -1, self.n_heads, self.d_state))
        C_blocks = jnp.reshape(self.C, (self.batch_size, -1, self.n_heads, self.d_state))
        
        A_blocks = jnp.moveaxis(A_blocks, -1, 1)
        A_cumsum = jnp.cumsum(A_blocks, axis=1)
        
        # 1. Compute diagonal block outputs
        L = jnp.exp(self.segsum(A_blocks))
        Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                            C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = jnp.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
        states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = jnp.zeros((self.batch_size, 1, self.d_state))
        states = jnp.concatenate([initial_states, states], axis=1)
        
        decay_chunk = jnp.exp(self.segsum(jnp.pad(A_cumsum[:, :, -1:], ((0, 0), (1, 0)))))
        new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]

def get_inputs():
    return [jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_length, n_heads, d_head))]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]
