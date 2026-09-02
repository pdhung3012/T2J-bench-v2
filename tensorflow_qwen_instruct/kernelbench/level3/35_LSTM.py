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
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=dropout, batch_input_shape=(None, sequence_length, input_size), unroll=False, go_backwards=False)
        self.fc = Dense(output_size)

    def call(self, x, initial_h=None, initial_c=None):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param initial_h: Optional initial hidden state (num_layers, batch_size, hidden_size)
        :param initial_c: Optional initial cell state (num_layers, batch_size, hidden_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        batch_size = tf.shape(x)[0]

        if initial_h is None:
            initial_h = tf.random.normal((self.num_layers, batch_size, self.hidden_size))
        if initial_c is None:
            initial_c = tf.random.normal((self.num_layers, batch_size, self.hidden_size))

        out, _, _ = self.lstm(x, initial_state=[initial_h, initial_c])  # out: (batch_size, seq_length, hidden_size)
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
    return [tf.random.normal((batch_size, sequence_length, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
