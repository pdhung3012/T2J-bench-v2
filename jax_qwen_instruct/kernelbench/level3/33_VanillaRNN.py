import jax
import jax.numpy as jnp

class Model:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = jnp.zeros((batch_size, hidden_size))
        
        # Define the RNN cell components (input to hidden, hidden to output)
        self.i2h = jax.nn.linear.Dense(hidden_size, input_size + hidden_size)  # Input to hidden
        self.h2o = jax.nn.linear.Dense(output_size, hidden_size)  # Hidden to output
        self.tanh = jax.nn.tanh  # Activation function for hidden state
    
    def forward(self, x: jnp.ndarray, initial_hidden=None) -> jnp.ndarray:
        """
        Forward pass of the Vanilla RNN.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
        """
        if initial_hidden is not None:
            self.hidden = initial_hidden
        combined = jnp.concatenate((x, self.hidden), axis=1)  # Concatenate input and hidden state
        self.hidden = self.tanh(self.i2h(combined))  # Update hidden state
        output = self.h2o(self.hidden)  # Compute output
        return output

batch_size = 256
input_size = 16384
hidden_size = 16384
output_size = 8192
sequence_length = 256

def get_inputs():
    return [jax.random.normal(key=jax.random.PRNGKey(0), shape=(batch_size, input_size)), jax.random.normal(key=jax.random.PRNGKey(1), shape=(batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
