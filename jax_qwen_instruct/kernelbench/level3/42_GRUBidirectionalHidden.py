import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
    
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return h_n

def get_inputs(batch_size, seq_len, input_size, hidden_size, num_layers):
    key = random.PRNGKey(0)
    rng_key, subkey = random.split(key)
    x = random.normal(subkey, (seq_len, batch_size, input_size))
    h0 = random.normal(subkey, (num_layers * 2, batch_size, hidden_size))
    return x, h0

def get_init_inputs(input_size, hidden_size, num_layers):
    return input_size, hidden_size, num_layers

batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

x, h0 = get_inputs(batch_size, seq_len, input_size, hidden_size, num_layers)
h_n = Model(get_init_inputs(input_size, hidden_size, num_layers)).apply({'params': {}}, x, h0)['h_n']
