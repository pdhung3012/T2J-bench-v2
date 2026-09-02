import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

class Model(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(Model, self).__init__()
        # Initialize hidden state with random values
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=dropout, batch_input_shape=(None, None, input_size), unroll=False)
        self.fc = Dense(output_size)
    
    def call(self, x, h0, c0):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: Initial hidden state, shape (num_layers * direction_num, batch_size, hidden_size)
        :param c0: Initial cell state, shape (num_layers * direction_num, batch_size, hidden_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        
        # Forward propagate LSTM
        out, _, _ = self.lstm(x, initial_state=[h0, c0])  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, output_size)
        
        return out

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [tf.random.normal((batch_size, sequence_length, input_size)),tf.random.normal((num_layers, batch_size, hidden_size)),tf.random.normal((num_layers, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
