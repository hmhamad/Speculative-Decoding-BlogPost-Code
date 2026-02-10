import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
target_model_name = "Qwen/Qwen3-4B-Instruct-2507"

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
def sample_from_probs(probs):
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1)

@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

@torch.no_grad()
def draft_k_tokens(input_ids, attention_mask, k):
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
def verify_and_accept(input_ids, attention_mask, drafted_tokens):
    """
    Returns:
        accepted_tokens: list of (1,1) tensors
        residual_probs: tensor or None
        target_logits: logits from target model over extended sequence
    """
    extended_ids = torch.cat([input_ids, drafted_tokens], dim=1)
    extended_mask = torch.cat(
        [attention_mask, torch.ones_like(drafted_tokens)],
        dim=1
    )

    draft_logits = forward_logits(draft_model, extended_ids, extended_mask)
    target_logits = forward_logits(target_model, extended_ids, extended_mask)

    accepted = []

    for i in range(drafted_tokens.size(1)):
        pos = input_ids.size(1) + i - 1
        token_id = drafted_tokens[0, i]

        draft_probs = torch.softmax(draft_logits[0, pos], dim=-1)
        target_probs = torch.softmax(target_logits[0, pos], dim=-1)

        p_draft = draft_probs[token_id]
        p_target = target_probs[token_id]

        alpha = torch.clamp(p_target / p_draft, max=1.0)

        if torch.rand(()) <= alpha:
            accepted.append(token_id.view(1, 1))
            if token_id.item() == eos_id:
                return accepted, None, target_logits
        else:
            residual = torch.clamp(target_probs - draft_probs, min=0.0)
            return accepted, residual, target_logits

    return accepted, None, target_logits

# ------------------------------------------------------------
# Main speculative decoding loop
# ------------------------------------------------------------
with torch.no_grad():
    generated = 0

    while generated < max_new_tokens:
        # 1. Draft K tokens
        drafted_tokens = draft_k_tokens(input_ids, attention_mask, K)

        # 2. Verify in parallel
        accepted_tokens, residual_probs, target_logits = verify_and_accept(
            input_ids, attention_mask, drafted_tokens
        )

        # 3. Append accepted tokens
        for tok in accepted_tokens:
            input_ids = torch.cat([input_ids, tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
                dim=1
            )

        # 4. Rejection or full acceptance
        if residual_probs is not None:
            # rejection → sample from residual
            next_token = sample_from_probs(residual_probs).view(1, 1)
        else:
            # full acceptance → reuse final target logits
            # Since the final token of the draft gives us the logits for the next token, if every drafted token is
            # accepted, we can sample from it normally. This gives us a maximum of 𝐾 + 1 tokens per loop,
            # over the naive implementation which would only return 𝐾 tokens
            pos = input_ids.size(1) - 1
            probs = torch.softmax(target_logits[0, pos] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).view(1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        generated = input_ids.shape[1] - orig_len

        if next_token.item() == eos_id:
            break

# ------------------------------------------------------------
# Decode
# ------------------------------------------------------------
generated_ids = input_ids[0, orig_len:]
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(output_text)
