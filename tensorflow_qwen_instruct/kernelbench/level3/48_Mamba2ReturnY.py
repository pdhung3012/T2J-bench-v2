import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
tfb = tfp.bijectors

class Model(tfk.Model):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = tf.Variable(tf.random.normal((batch_size, seq_length, n_heads)))
        self.B = tf.Variable(tf.random.normal((batch_size, seq_length, n_heads, d_state)))
        self.C = tf.Variable(tf.random.normal((batch_size, seq_length, n_heads, d_state)))
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = tf.shape(x)[-1]
        x_cumsum = tf.cumsum(x, axis=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)
        x_segsum = x_segsum * tf.cast(mask, x.dtype)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks = tf.reshape(X, (self.batch_size, -1, self.n_heads, self.d_head))
        A_blocks = tf.reshape(self.A, (self.batch_size, -1, self.n_heads))
        B_blocks = tf.reshape(self.B, (self.batch_size, -1, self.n_heads, self.d_state))
        C_blocks = tf.reshape(self.C, (self.batch_size, -1, self.n_heads, self.d_state))
        
        A_blocks = tf.transpose(A_blocks, perm=[0, 2, 1, 3])
        A_cumsum = tf.cumsum(A_blocks, axis=-1)
        
        # 1. Compute diagonal block outputs
        L = tf.math.exp(self.segsum(A_blocks))
        Y_diag = tf.tensordot(C_blocks, B_blocks, axes=[[2], [1]]) * L
        Y_diag = tf.reshape(Y_diag, (self.batch_size, -1, self.n_heads, self.d_head, self.d_head))
        
        # 2. Compute intra-chunk states
        decay_states = tf.math.exp(A_cumsum[:, :, -1:, :] - A_cumsum)
        states = tf.tensordot(B_blocks, decay_states, axes=[[2], [1]]) * tf.reshape(X_blocks, (self.batch_size, -1, self.n_heads, self.d_head))
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = tf.zeros((self.batch_size, 1, self.d_state))
        states = tf.concat([initial_states, states], axis=1)
        
        decay_chunk = tf.math.exp(self.segsum(tf.pad(A_cumsum[:, :, -1:], [[0, 0], [0, 1]])))
        new_states = tf.tensordot(decay_chunk, states, axes=[[1], [0]])
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = tf.math.exp(A_cumsum)
        Y_off = tf.tensordot(C_blocks, states, axes=[[2], [1]]) * state_decay_out
        Y_off = tf.reshape(Y_off, (self.batch_size, -1, self.n_heads, self.d_head, self.d_head))
        
        # Combine diagonal and off-diagonal terms
        Y = Y_diag + Y_off
        
        return Y

# Test parameters
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [tf.random.normal((batch_size, seq_length, n_heads, d_head))]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]
