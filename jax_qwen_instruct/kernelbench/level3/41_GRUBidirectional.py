import jax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import normal

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)

    def init_params(self, key, batch_size, num_layers, hidden_size):
        w_init = normal(key, (self.gru.weight_ih.shape[0], self.gru.weight_ih.shape[1]), dtype=jnp.float32)
        b_init = normal(key, (self.gru.weight_ih.shape[1],), dtype=jnp.float32)
        h0 = jnp.zeros((num_layers * 2, batch_size, hidden_size))
        return {'w_ih': w_init, 'b_ih': b_init, 'h0': h0}

    def init(self, key, batch_size, num_layers, hidden_size):
        params = self.init_params(key, batch_size, num_layers, hidden_size)
        return Model(**params)

    def __call__(self, x, h0):
        params = self.params
        w_ih, b_ih, h0 = params['w_ih'], params['b_ih'], params['h0']
        x = jnp.swapaxes(x, 0, 1)  # Reshape to (batch_size, seq_len, input_size)
        h0 = jnp.swapaxes(h0, 0, 1)  # Reshape to (num_layers * num_directions, batch_size, hidden_size)
        output, _ = self.gru(x, h0)
        return jnp.swapaxes(output, 0, 1), h0

def get_inputs():
    key = random.PRNGKey(0)
    batch_size = 10
    seq_len = 512
    input_size = 128
    hidden_size = 256
    num_layers = 6
    x = random.normal(key, (seq_len, batch_size, input_size))
    h0 = random.normal(key, (num_layers * 2, batch_size, hidden_size))
    return x, h0

def get_init_inputs():
    input_size = 128
    hidden_size = 256
    num_layers = 6
    return input_size, hidden_size, num_layers

# Test code
x, h0 = get_inputs()
model = Model(*get_init_inputs())
output, h_n = model(x, h0)
print(output.shape, h_n.shape)
