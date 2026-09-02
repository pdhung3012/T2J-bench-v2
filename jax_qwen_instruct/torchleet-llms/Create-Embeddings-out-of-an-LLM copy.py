import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Amazon Reviews dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
reviews = dataset['full'][:1000] # first 1000 reviews

# Load SmolLM2-135M model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Tokenize the reviews with padding for batch processing
encodings = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# 4. Forward pass with output_hidden_states=True to get all hidden states
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

# 5. Extract last hidden states (batch_size, seq_len, hidden_dim)
last_hidden_states = outputs.hidden_states[-1]

# 6. Compute sentence embeddings by averaging token embeddings excluding padding tokens
# attention_mask has 1 for real tokens, 0 for padding
expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.shape)
sum_embeddings = torch.sum(last_hidden_states * expanded_mask, dim=1)
sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)  # avoid division by zero
sentence_embeddings = sum_embeddings / sum_mask

print("Sentence embeddings shape:", sentence_embeddings.shape)  # (10, hidden_dim)

# --- Cosine similarity for a given keyword ---

# Example keyword
keyword = "quality"

# Tokenize and embed the keyword the same way
keyword_enc = tokenizer(keyword, return_tensors="pt")
keyword_input_ids = keyword_enc['input_ids']
keyword_attention_mask = keyword_enc['attention_mask']

with torch.no_grad():
    keyword_outputs = model(keyword_input_ids, attention_mask=keyword_attention_mask, output_hidden_states=True)

keyword_last_hidden = keyword_outputs.hidden_states[-1]
keyword_mask = keyword_attention_mask.unsqueeze(-1).expand(keyword_last_hidden.shape)
keyword_embedding = (keyword_last_hidden * keyword_mask).sum(dim=1) / torch.clamp(keyword_mask.sum(dim=1), min=1e-9)

# Compute cosine similarity between keyword embedding and each review embedding
cosine_similarities = torch.nn.functional.cosine_similarity(sentence_embeddings, keyword_embedding)

for i, (review, sim) in enumerate(zip(reviews, cosine_similarities)):
    print(f"\nReview #{i+1} similarity to '{keyword}': {sim.item():.4f}")
    print(review)
