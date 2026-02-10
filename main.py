import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
temperature = 0

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

model_draft = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    dtype='auto',
    device_map='mps'
)
model_large = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    dtype='auto',
    device_map='mps'
)

device = model_draft.device

prompts = [
    'List the top 10 countries by population in decreasing order. Return the list only, no commentary.',
]

messages = [
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": p},
    ]
    for p in prompts
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, padding=True, return_tensors='pt').to(device)

input_ids = model_inputs.input_ids          # B x L
attention_mask = model_inputs.attention_mask

orig_len = input_ids.shape[1]
eos_id = tokenizer.eos_token_id
batch_size = input_ids.size(0)

finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

def sample_next_token(next_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    next_logits: (B, V)
    returns next_token_ids: (B, 1) long
    """
    if temperature and temperature > 0:
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token_ids = torch.multinomial(probs, 1)  # (B, 1)
    else:
        next_token_ids = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)
    return next_token_ids.long()

@torch.no_grad()
def decode_one_token_step(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    finished: torch.Tensor,
    eos_id: int,
    temperature: float,
):
    """
    Appends exactly one token per batch element (or eos if finished).
    Returns updated (input_ids, attention_mask, finished).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)
    next_logits = logits[:, -1, :]  # (B, V)

    next_token_ids = sample_next_token(next_logits, temperature)  # (B, 1)

    # mark newly finished
    is_eos = next_token_ids.squeeze(1) == eos_id
    finished = finished | is_eos

    # force eos for already-finished sequences
    next_token_ids = next_token_ids.clone()
    next_token_ids[finished] = eos_id

    # append
    input_ids = torch.cat([input_ids, next_token_ids], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1
    )
    return input_ids, attention_mask, finished, next_logits

with torch.no_grad():
    while not finished.all():
        input_ids, attention_mask, finished, next_logits = decode_one_token_step(
            model=model_draft,
            input_ids=input_ids,
            attention_mask=attention_mask,
            finished=finished,
            eos_id=eos_id,
            temperature=temperature,
        )

responses = tokenizer.batch_decode(input_ids[:, orig_len:], skip_special_tokens=True)
for i, res in enumerate(responses):
    print(f"Response {i+1}: {res}\n")
