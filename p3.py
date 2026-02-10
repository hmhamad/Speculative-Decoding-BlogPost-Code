# In Program 2, we learned how to generate tokens in chunks of size K. We'll now give that behavior a name: drafting.

# The key idea of speculative decoding is simple:

# a small, fast draft model proposes several tokens

# a larger, more accurate target model verifies them

# we keep every token the target model agrees with

# at the first disagreement, the target model takes over

# Crucially, the target model produces exactly the same distribution as if it were decoding alone

# High-level algorithm 

# Draft model samples K tokens

# Target model runs one forward pass on the extended sequence

# Tokens are accepted until the first mismatch

# Repeat

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
target_model_name = "Qwen/Qwen2.5-1.8B-Instruct"

device = "mps"
temperature = 0.8
max_new_tokens = 100
K = 4

# ------------------------------------------------------------
# Load tokenizer and models
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    device_map=device,
    torch_dtype="auto"
)

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    device_map=device,
    torch_dtype="auto"
)

eos_id = tokenizer.eos_token_id

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

inputs = tokenizer(text, return_tensors="pt").to(draft_model.device)

input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
orig_len = input_ids.shape[1]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sample_next_token(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    """Returns logits for all positions. shape: (1, seq_len, vocab)"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits

@torch.no_grad()
def draft_k_tokens(input_ids, attention_mask, k):
    """Draft K tokens using the small model."""
    tokens = []
    for _ in range(k):
        logits = forward_logits(draft_model, input_ids, attention_mask)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens.append(next_token)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )
        if next_token.item() == eos_id:
            break
    return torch.cat(tokens, dim=1)

@torch.no_grad()
def verify_drafted_tokens(input_ids, attention_mask, drafted_token_ids):
    """
    Implements speculative verification.
    Returns: accepted count, residual_probs, target_logits
    """
    # Run both models on the extended sequence
    extended_ids = torch.cat([input_ids, drafted_token_ids], dim=1)
    extended_mask = torch.cat(
        [attention_mask, torch.ones_like(drafted_token_ids)],
        dim=1
    )

    draft_logits = forward_logits(draft_model, extended_ids, extended_mask)
    target_logits = forward_logits(target_model, extended_ids, extended_mask)

    accepted = 0
    residual_probs = None

    # Token-by-token acceptance test
    for i in range(drafted_token_ids.size(1)):
        pos = input_ids.size(1) + i - 1
        token_id = drafted_token_ids[0, i]

        p_draft = torch.softmax(draft_logits[0, pos], dim=-1)[token_id]
        p_target = torch.softmax(target_logits[0, pos], dim=-1)[token_id]

        alpha = torch.clamp(p_target / p_draft, max=1.0)

        if torch.rand(()) <= alpha:
            accepted += 1
            if token_id.item() == eos_id:
                break
        else:
            # Compute residual distribution
            target_probs = torch.softmax(target_logits[0, pos], dim=-1)
            draft_probs = torch.softmax(draft_logits[0, pos], dim=-1)
            residual_probs = torch.clamp(target_probs - draft_probs, min=0.0)
            break

    return accepted, residual_probs, target_logits

# ------------------------------------------------------------
# Main loop (simplified - demonstrates verification)
# ------------------------------------------------------------
with torch.no_grad():
    # Draft K tokens
    drafted_tokens = draft_k_tokens(input_ids, attention_mask, K)
    print(f"Drafted {drafted_tokens.shape[1]} tokens: {tokenizer.decode(drafted_tokens[0])}")

    # Verify them
    accepted, residual_probs, target_logits = verify_drafted_tokens(
        input_ids, attention_mask, drafted_tokens
    )

    print(f"Accepted: {accepted} / {drafted_tokens.shape[1]}")
    if residual_probs is not None:
        print("Rejection occurred - would sample from residual distribution")
    else:
        print("All tokens accepted!")
