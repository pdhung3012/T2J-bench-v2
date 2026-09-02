import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_text as text

# Load Amazon Reviews dataset
from datasets import load_dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
reviews = dataset['full'][:1000] # first 1000 reviews

# Load SmolLM2-135M model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(isinstance(model, tf.keras.Model))  # Should print: True

device = "cuda" if tf.config.experimental.list_physical_devices('GPU') else "cpu"
model = model.load_weights(tf.convert_to_tensor(model.state_dict())).eval()

device = tf.device(device)
model = model.to(device)

# 3. Tokenize the reviews with padding for batch processing
encodings = tokenizer(reviews, return_tensors="tf", padding=True, truncation=True)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# 4. Forward pass with output_hidden_states=True to get all hidden states
with tf.name_scope("forward_pass"):
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

# 5. Extract last hidden states (batch_size, seq_len, hidden_dim)
last_hidden_states = outputs.hidden_states[-1]

# 6. Compute sentence embeddings by averaging token embeddings excluding padding tokens
# attention_mask has 1 for real tokens, 0 for padding
expanded_mask = tf.cast(attention_mask, tf.float32)
sum_embeddings = tf.reduce_sum(last_hidden_states * expanded_mask[:, :, None], axis=1)
sum_mask = tf.reduce_sum(expanded_mask, axis=1, keepdims=True)
sum_mask = tf.maximum(sum_mask, tf.ones_like(sum_mask) * 1e-9)  # avoid division by zero
sentence_embeddings = sum_embeddings / sum_mask

print("Sentence embeddings shape:", sentence_embeddings.shape)  # (10, hidden_dim)

# --- Cosine similarity for a given keyword ---

# Example keyword
keyword = "quality"

# Tokenize and embed the keyword the same way
keyword_enc = tokenizer(keyword, return_tensors="tf")
keyword_input_ids = keyword_enc['input_ids']
keyword_attention_mask = keyword_enc['attention_mask']

with tf.name_scope("keyword_embedding"):
    keyword_outputs = model(keyword_input_ids, attention_mask=keyword_attention_mask, output_hidden_states=True)

keyword_last_hidden = keyword_outputs.hidden_states[-1]
keyword_mask = tf.cast(keyword_attention_mask, tf.float32)
keyword_embedding = tf.reduce_sum(keyword_last_hidden * keyword_mask[:, :, None], axis=1) / tf.reduce_sum(keyword_mask, axis=1, keepdims=True)
keyword_embedding = tf.maximum(keyword_embedding, tf.ones_like(keyword_embedding) * 1e-9)  # avoid division by zero

# Compute cosine similarity between keyword embedding and each review embedding
cosine_similarities = tf.keras.losses.CosineSimilarity(axis=-1)(sentence_embeddings, keyword_embedding)

for i, (review, sim) in enumerate(zip(reviews, cosine_similarities)):
    print(f"\nReview #{i+1} similarity to '{keyword}': {sim.numpy().item():.4f}")
    print(review.numpy())
