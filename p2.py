# In Program 1, we generated text one token at a time. That's conceptually simple, but structurally limiting: everything is hard-coded around a single forward pass producing a single token.

# Before we can talk about speculative decoding, we need one more refactor. Instead of generating exactly one token per loop, we'll generalize our code to generate K tokens at a time. Importantly, we're still using a single model, and we're not changing the decoding logic — just packaging it differently.

# This step may feel trivial, but it's the key that lets us later “draft ahead.”

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps"
temperature = 0.8
max_new_tokens = 100
K = 4   # number of tokens generated per iteration

# ------------------------------------------------------------
# Load model and tokenizer
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype="auto"
)

# ------------------------------------------------------------
# Prompt
# ------------------------------------------------------------
prompt = "List the top 5 countries by population in decreasing order."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# ------------------------------------------------------------
# Tokenize
# ------------------------------------------------------------
inputs = tokenizer(text, return_tensors="pt").to(model.device)

input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
orig_len = input_ids.shape[1]

eos_id = tokenizer.eos_token_id

# ------------------------------------------------------------
# Helper: sample next token
# ------------------------------------------------------------
def sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (1, vocab_size)
    returns: (1, 1)
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# ------------------------------------------------------------
# Helper 1: forward pass
# ------------------------------------------------------------
@torch.no_grad()
def forward_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Runs the model and returns logits for the last token.

    returns:
        next_logits: (1, vocab_size)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    return outputs.logits[:, -1, :]

# ------------------------------------------------------------
# Helper 2: generate K tokens
# ------------------------------------------------------------
@torch.no_grad()
def generate_k_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
    temperature: float,
    eos_id: int,
):
    """
    Generates up to k tokens sequentially.

    returns:
        updated input_ids
        updated attention_mask
        finished (bool)
    """
    finished = False

    for _ in range(k):
        next_logits = forward_pass(model, input_ids, attention_mask)
        next_token_id = sample_next_token(next_logits, temperature)

        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1
        )

        if next_token_id.item() == eos_id:
            finished = True
            break

    return input_ids, attention_mask, finished

# ------------------------------------------------------------
# Main decoding loop
# ------------------------------------------------------------
with torch.no_grad():
    finished = False
    generated = 0

    while not finished and generated < max_new_tokens:
        input_ids, attention_mask, finished = generate_k_tokens(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            k=K,
            temperature=temperature,
            eos_id=eos_id,
        )
        generated = input_ids.shape[1] - orig_len

# ------------------------------------------------------------
# Decode only generated tokens
# ------------------------------------------------------------
generated_ids = input_ids[0, orig_len:]
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(output_text)
