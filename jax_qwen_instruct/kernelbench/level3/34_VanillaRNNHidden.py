import jax
import jax.numpy as jnp
from jax import vmap

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Dense(input_size + hidden_size, dtype=jnp.float32)  # Input to hidden
        self.h2o = nn.Dense(hidden_size, dtype=jnp.float32)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    @vmap
    def forward(self, x, h0):
        seq_len, batch_size, _ = x.shape
        hidden = h0
        outputs = []

        for t in range(seq_len):
            combined = jnp.concatenate((x[t], hidden), axis=0)  # Concatenate input and hidden state
            hidden = self.tanh(self.i2h(combined))  # Update hidden state
            output = self.h2o(hidden)  # Compute output
            outputs.append(output)

        return jnp.stack(outputs, axis=0)  # (seq_len, batch_size, output_size)

# === Test configuration ===
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [
        jax.random.normal(key=jax.random.PRNGKey(0), shape=(sequence_length, batch_size, input_size)),
        jax.random.normal(key=jax.random.PRNGKey(1), shape=(batch_size, hidden_size))
    ]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
