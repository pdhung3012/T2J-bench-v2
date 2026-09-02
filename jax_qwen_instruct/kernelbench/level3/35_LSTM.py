import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.shape[0]

        if h0 is None:
            h0 = random.normal(random.PRNGKey(0), (self.num_layers, batch_size, self.hidden_size))
        if c0 is None:
            c0 = random.normal(random.PRNGKey(1), (self.num_layers, batch_size, self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])     # out: (batch_size, output_size)

        return out

# === Test configuration ===
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [random.normal(random.PRNGKey(2), (batch_size, sequence_length, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
