import jax
import jax.numpy as jnp
from jax import random

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
    
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return h_n

def get_inputs(batch_size=10, seq_len=512, input_size=128, hidden_size=256, num_layers=6):
    key = random.PRNGKey(0)
    rng, subkey = random.split(key)
    x = random.normal(subkey, (seq_len, batch_size, input_size))
    h0 = random.normal(subkey, (num_layers * 1, batch_size, hidden_size))
    return x, h0

def get_init_inputs():
    return [128, 256, 6]

x, h0 = get_inputs()
h_n = Model().apply({'params': {}}, x, h0)
h_n
