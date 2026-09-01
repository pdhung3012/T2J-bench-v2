import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "qwen-instruct": "/home/hungphd/git/Qwen2.5-3B-Instruct",
    "qwen": "Qwen/Qwen2.5-Coder-32B",
    "opencode": "m-a-p/OpenCodeInterpreter-DS-33B",
    "gemma": "google/gemma-3-12b-it",
    "nxcode": "NTQAI/Nxcode-CQ-7B-orpo",
    "phi": "microsoft/Phi-4-mini-instruct",
}


def load_llm(name):
    model_id = MODELS[name]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    return tokenizer, model


def ask(tokenizer, model, question):
    messages = [
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

    answer = output[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(
        answer,
        skip_special_tokens=True
    )


# ------------------------------------
# Example
# ------------------------------------

tokenizer, model = load_llm("qwen-instruct")

question = "Translate this following Python code to JAX code. Return the result as python code only, without any explanation or comments."

answer = ask(tokenizer, model, question)

print(answer)