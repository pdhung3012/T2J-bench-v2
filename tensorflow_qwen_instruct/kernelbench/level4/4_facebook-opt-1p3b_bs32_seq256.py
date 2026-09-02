import tensorflow as tf
from transformers import TFAutoModelForCausalLM, TFAutoConfig

class Model(tf.keras.Model):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = TFAutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)

    def call(self, x):
        return self.model(x).logits

model_name = "facebook/opt-1.3b"
config = TFAutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 32

def get_inputs():
    inputs = tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    return [inputs]

def get_init_inputs():
    return [model_name, config]
