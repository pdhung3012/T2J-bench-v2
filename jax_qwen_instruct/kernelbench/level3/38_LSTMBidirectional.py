import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_inputs(rng_key):
    batch_size = 10
    sequence_length = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6
    output_size = 10
    dropout = 0.0
    
    x = random.normal(key=random.PRNGKey(0), shape=(batch_size, sequence_length, input_size))
    h0 = random.normal(key=random.PRNGKey(1), shape=(num_layers * 2, batch_size, hidden_size))
    c0 = random.normal(key=random.PRNGKey(2), shape=(num_layers * 2, batch_size, hidden_size))
    return x, h0, c0

def get_init_inputs():
    return [128, 256, 6, 10, 0.0]
