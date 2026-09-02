import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        out, state = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return state[0]

def get_inputs(seed=jax.random.PRNGKey(0)):
    key, subkey = random.split(seed)
    x = random.normal(subkey, (batch_size, sequence_length, input_size))
    h0 = random.normal(subkey, (num_layers, batch_size, hidden_size))
    c0 = random.normal(subkey, (num_layers, batch_size, hidden_size))
    return [x, h0, c0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
