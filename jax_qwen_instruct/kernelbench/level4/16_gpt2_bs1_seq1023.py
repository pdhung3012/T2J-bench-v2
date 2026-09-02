import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM, AutoConfig

class Model(jax.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)

    @jax.jit
    def __call__(self, x):
        return self.model(x).logits

model_name = "gpt2"
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 1023
batch_size = 1

def get_inputs():
    inputs = jax.random.randint(next_rng(), (batch_size, sequence_length), 0, vocab_size)
    return [inputs]

def get_init_inputs():
    return [model_name, config]
