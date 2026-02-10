# So far, we've relied entirely on model.generate, which hides the entire decoding loop behind a single function call. That's great for production, but not great for learning.

# If we want to understand — and eventually re-implement — speculative decoding, we need to peel back that abstraction. At its core, text generation is nothing more than a loop that repeatedly:

# runs the model on the current tokens

# looks at the logits for the last position

# samples the next token

# appends it to the input

# In this program, we'll re-implement exactly that logic, but restricted to generating one token per step. This is the smallest unit we need to fully control decoding.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps"
temperature = 0.8
max_new_tokens = 100

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
    returns: (1, 1) next token id
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id

# ------------------------------------------------------------
# Manual decoding loop (one token at a time)
# ------------------------------------------------------------
with torch.no_grad():
    for step in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # logits: (1, seq_len, vocab_size)
        next_logits = outputs.logits[:, -1, :]

        next_token_id = sample_next_token(next_logits, temperature)

        # append token
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1
        )

        # stop if EOS
        if next_token_id.item() == eos_id:
            break

# ------------------------------------------------------------
# Decode only generated tokens
# ------------------------------------------------------------
generated_ids = input_ids[0, orig_len:]
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(output_text)
