import jax
import jax.numpy as jnp
from flax.linen import Module, dense
from flax.training import train_state
import optax

class Decoder(Module):
    vocab_size: int
    hidden_size: int

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dense = dense(self.vocab_size, name="output_layer")

    @jax.jit
    def __call__(self, decoder_input, encoder_outputs, hidden_state, cell_state):
        attention_scores = jnp.dot(encoder_outputs, hidden_state)
        attention_weights = jax.nn.softmax(attention_scores)
        context_vector = jnp.dot(attention_weights, encoder_outputs)

        hidden_state = self.apply("update_hidden_state", hidden_state, context_vector)
        output = self.dense(context_vector)

        return output, hidden_state, cell_state

    @staticmethod
    @jax.jit
    def apply(name, hidden_state, context_vector):
        return hidden_state + context_vector

def main():
    vocab_size = 10000
    hidden_size = 256
    tgt_seq_length = 10

    decoder = Decoder(vocab_size=vocab_size, hidden_size=hidden_size)
    hidden_state = jnp.zeros((1, hidden_size))
    cell_state = jnp.zeros((1, hidden_size))
    decoder_input = jnp.zeros((1, vocab_size))
    encoder_outputs = jnp.zeros((1, tgt_seq_length, hidden_size))

    output_sequence = []

    for _ in range(tgt_seq_length):
        output, hidden_state, cell_state = decoder(decoder_input, encoder_outputs, hidden_state, cell_state)
        predicted = jnp.argmax(output, axis=1)
        output_sequence.append(predicted.item())

        decoder_input = jax.nn.one_hot(predicted, vocab_size)

    print(f"Input: {jnp.zeros((1, vocab_size)).tolist()}, Output: {output_sequence}")

if __name__ == "__main__":
    main()
