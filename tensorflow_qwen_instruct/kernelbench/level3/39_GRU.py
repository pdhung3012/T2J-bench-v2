import tensorflow as tf
from tensorflow.keras.layers import GRU

class Model(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(Model, self).__init__()
        
        self.gru = GRU(input_size, hidden_size, num_layers, bias=bias, return_sequences=True, return_state=True, dropout=0, unroll=False, go_backwards=False)
    
    def call(self, x, h0=None):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        output, h_n = self.gru(x, initial_state=h0)
        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [tf.random.normal([seq_len, batch_size, input_size]),tf.random.normal([num_layers, batch_size, hidden_size])]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
