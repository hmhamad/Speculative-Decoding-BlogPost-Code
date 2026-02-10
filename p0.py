import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" # use "cuda" if you have an NVIDIA GPU
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
orig_len = input_ids.shape[1]

# ------------------------------------------------------------
# Generate (sampling, not greedy)
# ------------------------------------------------------------
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )

# ------------------------------------------------------------
# Decode only the generated continuation
# ------------------------------------------------------------
generated_ids = output_ids[0, orig_len:]
output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(output_text)
