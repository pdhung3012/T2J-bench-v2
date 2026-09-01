import math
import torch
from torch import nn

# Helper function to rotate the last half of a tensor
# Used in rotary positional embeddings to compute sine and cosine rotations
def rotate_half(x):
    # Split tensor into two halves along the last dimension
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Rotate: negate the second half and concatenate it with the first half
    return torch.cat((-x2, x1), dim=-1)

# Applies rotary positional embeddings to query (q) and key (k) tensors
# Uses sine and cosine positional encodings to enhance positional awareness
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Expand cos and sin tensors for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotations to q and k using cos and sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Repeats key-value tensors for multiple attention heads
# Ensures compatibility between the number of attention heads and key-value heads
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # Expand the number of key-value heads by repeating them
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # Reshape to align with the expected multi-head attention format
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Computes rotary positional embeddings for queries and keys
class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        # Precompute frequency for sine/cosine embeddings
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    @torch.no_grad()
    def forward(self, x):
        # Generate positions (sequence indices) for the input
        pos = torch.arange(x.shape[-2], dtype=torch.long)
        # Compute angles for sine and cosine embeddings
        angles = torch.einsum("p,f->pf", pos.float(), self.freq).unsqueeze(dim=0)
        # Duplicate angles for sine and cosine embeddings
        emb = torch.cat((angles, angles), dim=-1)
        # Return cosine and sine components of the positional embeddings
        return emb.cos(), emb.sin()

# Implements attention with rotary positional embeddings
class RopeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Model dimensions and attention configurations
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads  # Number of key-value heads
        self.rope_theta = 10000.0  # Scaling factor for rotary embeddings

        # Linear projections for queries, keys, values, and output
        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Rotary embedding generator
        self.rotary_emb = RotaryEmbedder(base=self.rope_theta, dim=self.head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        # Input dimensions: (batch_size, seq_len, hidden_size)
        b, q, _ = hidden_states.size()

        # Project input hidden states into queries, keys, and values
        q_states = self.W_query(hidden_states)
        k_states = self.W_key(hidden_states)
        v_states = self.W_value(hidden_states)

        # Reshape and transpose for multi-head attention
        q_states = q_states.view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)

        # Compute rotary positional embeddings
        cos, sin = self.rotary_emb(q_states)
        # Apply positional embeddings to queries and keys
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # Repeat key and value tensors to match the number of query heads
        __kv_groups = self.num_heads // self.kv_heads
        k_states = repeat_kv(k_states, __kv_groups)
        v_states = repeat_kv(v_states, __kv_groups)

        # Compute attention scores (scaled dot-product attention)
        attn_weights = torch.matmul(q_states, k_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Add attention mask (e.g., for causal or padding masking)
        attn_weights = attn_weights + attention_mask

        # Normalize attention weights using softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Apply dropout to attention weights
        attn_weights = nn.functional.dropout(attn_weights, 0)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v_states)
        # Reshape and transpose back to original format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(b, q, -1)

        # Project the attention output back to the hidden size
        attn_output = self.W_output(attn_output)

        # Return the final attention output
        return attn_output
import torch
from torch import nn

# Multi-Layer Perceptron (MLP) class
class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        """
        Initialize an MLP with a gating mechanism and non-linear activation.

        Args:
            hidden_size (int): The size of the input and output embeddings.
            intermediate_size (int): The size of the hidden layer.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Linear layers for the gated MLP structure
        self.W_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.W_down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Activation function (SiLU)
        self.act_fn = torch.nn.modules.activation.SiLU()

    def forward(self, x):
        """
        Forward pass for the gated MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Apply gating mechanism and project back to hidden size
        down_proj = self.W_down(self.act_fn(self.W_gate(x)) * self.W_up(x))
        return down_proj


# RMSNorm class (Root Mean Square Normalization)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize RMSNorm.

        Args:
            hidden_size (int): The size of the input embeddings.
            eps (float): A small value to prevent division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # Learnable scaling factor
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Forward pass for RMSNorm.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate variance along the last dimension (hidden size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        # Normalize and scale
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# Rotary Embedder class
class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        """
        Initialize rotary embeddings for positional encodings.

        Args:
            dim (int): Dimensionality of embeddings (half the hidden size per head).
            base (float): Base frequency for rotary embeddings.
        """
        super().__init__()
        # Frequency for rotary embeddings
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def forward(self, x):
        """
        Compute cosine and sine embeddings for rotary position encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings.
        """
        pos = torch.arange(x.shape[-2], dtype=torch.float32)  # Sequence positions

        # Calculate angular frequencies
        angles = torch.einsum("p,f->pf", pos, self.freq).unsqueeze(0)
        
        # Create cosine and sine embeddings
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()


# LlamaDecoder class
class LlamaDecoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the LlamaDecoder layer with attention and MLP sublayers.

        Args:
            config: Configuration object containing model parameters.
        """
        super().__init__()
        # Attention mechanism with rotary embeddings
        self.self_attn = RopeAttention(config)

        # Feedforward neural network (MLP)
        self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

        # Pre-layer normalization (RMSNorm) for attention and MLP
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.pre_mlp_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, hidden_states, attention_mask):
        """
        Forward pass for the LlamaDecoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (torch.Tensor): Mask to prevent attention to certain positions.

        Returns:
            Tuple[torch.Tensor]: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Residual connection for attention sublayer
        residual = hidden_states

        # Apply RMSNorm before attention
        hidden_states = self.pre_attn_rmsnorm(hidden_states)

        # Generate a triangular attention mask (causal masking)
        attention_mask = torch.triu(torch.full((attention_mask.shape[-1], attention_mask.shape[-1]),
                                               fill_value=float('-inf')), diagonal=1)

        # Apply self-attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        # Add residual connection
        hidden_states += residual

        # Residual connection for MLP sublayer
        residual = hidden_states

        # Apply RMSNorm before MLP
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)

        # Pass through MLP
        hidden_states = self.mlp(hidden_states)

        # Add residual connection
        hidden_states += residual

        # Return the output hidden states
        return hidden_states,
# The main model containing the embedding layer, decoder stack, and normalization layer
class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Token embedding layer: maps input tokens to dense representations
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        # Stack of decoder layers (LlamaDecoder) defined by the configuration
        self.layers = nn.ModuleList([
            LlamaDecoder(config) for _ in range(config.num_hidden_layers)
        ])
        # RMSNorm: final layer normalization applied to hidden states
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        # Convert input token IDs to dense embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Pass embeddings through each decoder layer in the stack
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,  # Pass the attention mask
            )
            # Update hidden states with the output of the current decoder layer
            hidden_states = layer_outputs[0]

        # Apply final layer normalization to the hidden states
        hidden_states = self.norm(hidden_states)

        # Return the processed hidden states
        return hidden_states


# The complete language model, combining smolModel and a language modeling head (lm_head)
class smolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Core model containing embeddings and decoder stack
        self.model = smolModel(config)
        # Language modeling head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between the embedding layer and lm_head
        self.tie_weights()

    def tie_weights(self):
        # Ensures the lm_head shares weights with the embedding layer
        # This is a common optimization for language models
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the core model to obtain hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Obtain hidden states from the model's output
        hidden_states = outputs

        # Pass hidden states through the language modeling head
        logits = self.lm_head(hidden_states)
        # Ensure logits are returned in float for numerical stability
        logits = logits.float()

        # Return the output as a dictionary containing logits
        return {'logits': logits}


from transformers import AutoTokenizer, AutoModelForCausalLM


# Libraries
import torch
import torch.nn.functional as F
from torch import nn
import math

########################## HELPER FUNCTIONS ######################

def __generate(model, inputs, num_tokens, tokenizer, max_length=50):
    collect = []
    for _ in range(num_tokens):
        output = model(**inputs)
        output_id = torch.argmax(output['logits'][0, -1]).item()
        collect.append(output_id)
        if output_id == tokenizer.eos_token_id or len(collect) >= max_length:
            break
        # Update input_ids and attention_mask
        new_token = torch.tensor([output_id], device=inputs['input_ids'].device)
        inputs['input_ids'] = torch.cat([inputs['input_ids'][0], new_token]).unsqueeze(0)
        inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, 1), value=1)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(collect))


def check_solution(prompt, num_tokens, model_A, model_B, tokenizer, max_length=50):
    print(f"{'>'*20}\n\tPrompt\n{'<'*20}\n{prompt}\n\n")
    
    model_inputs = tokenizer(prompt, return_tensors='pt')
    
    try:
        print(f"{'>'*30}\n\tModel_A Generation\n{'<'*30}")
        print(__generate(model_A, model_inputs, num_tokens, tokenizer, max_length))
    except Exception as e:
        print(f"Error with Model_A: {e}")
    
    try:
        model_inputs = tokenizer(prompt, return_tensors='pt')
        print(f"\n\n{'>'*30}\n\tModel_B Generation\n{'<'*30}")
        print(__generate(model_B, model_inputs, num_tokens, tokenizer, max_length))
    except Exception as e:
        print(f"Error with Model_B: {e}")


class smolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3

import torch


# Load tokenizer and reference model
checkpoint = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
reference_model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Initialize smolLM
config = smolConfig()
test_model = smolLM(config)

# Load weights
state_dict = torch.load("../../temp/BareBones_SmolLM-135M.pt")
test_model.load_state_dict(state_dict, strict=False)

check_solution(prompt="Given the following film movie by a critic, rate it out of 10. Respond in a single number.\n\nThe movie started off extremely well, but just got worse after that.\nThe storyline was all over the place and everyone acted terribly.\n 10/10 would not recommend! \n\n ",
               num_tokens=1,
               model_A=reference_model,
               model_B=test_model, tokenizer=tokenizer)