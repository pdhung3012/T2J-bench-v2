import torch
import torch.nn.functional as F
from torch import nn

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
    return tokenizer.decode(collect)

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

class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def forward(self, x):
        pos = torch.arange(x.shape[-2], dtype=torch.float32)
        angles = torch.einsum("p,f->pf", pos, self.freq).unsqueeze(0)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.W_down(self.act_fn(self.W_gate(x) * self.W_up(x)))
        return down_proj

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def forward(self, x):
        pos = torch.arange(x.shape[-2], dtype=torch.float32)
        angles = torch.einsum("p,f->pf", pos, self.freq).unsqueeze(0)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()

class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)
        self.pre_mlp_rmsnorm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        attention_mask = torch.triu(torch.full((attention_mask.shape[-1], attention_mask.shape[-1]), fill_value=float('-inf')), diagonal=1)
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)[0]
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states,

class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states

class smolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = smolModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        return {'logits': logits}

from transformers import AutoTokenizer, AutoModelForCausalLM

# Libraries
import torch
import torch.nn.functional as F
from torch import nn
import math

class smolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3

class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-05)

    def forward(self, input_ids=None, attention_mask=None):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states

class smolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = smolModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        return {'logits'}

from transformers import AutoTokenizer, AutoModelForCausalLM

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
