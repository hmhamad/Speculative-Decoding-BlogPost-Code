# Program 6: Batched speculative decoding benchmarks and plots
# This script benchmarks target-only vs speculative decoding across batch sizes and K,
# then plots speedup vs K and speedup vs batch size.

import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Configuration ----------------
draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
target_model_name = "Qwen/Qwen3-4B-Instruct-2507"
device = "mps"
temperature = 0.8
max_new_tokens = 128

Ks = [1, 2, 4, 8, 16]
batch_sizes = [1, 2, 4, 8, 16]

num_runs = 1  # keep small for demo/runtime

# ---------------- Prompts ----------------
prompts = [
    "List the top 10 countries by population in decreasing order.",
    "Explain the difference between TCP and UDP.",
    "Give three reasons why the sky appears blue.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are the main causes of climate change?",
    "Explain gradient descent in simple terms.",
    "List five famous physicists and their contributions.",
    "What is speculative decoding in language models?",
    "Explain the concept of overfitting in machine learning.",
    "Describe how a transformer model works.",
    "List the planets in our solar system with one fact each.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain backpropagation at a high level.",
    "Give an example of a greedy algorithm.",
    "Explain the bias-variance tradeoff.",
    "What is a neural network activation function?",
    "Describe the role of attention in transformers.",
    "Explain the difference between precision and recall.",
    "What is batching and why is it useful in ML?",
    "Explain why GPUs are good for deep learning."
]

# ---------------- Load tokenizer and models ----------------
tokenizer = AutoTokenizer.from_pretrained(draft_model_name)

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name, device_map=device, torch_dtype="auto"
)
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name, device_map=device, torch_dtype="auto"
)

eos_id = tokenizer.eos_token_id

# ---------------- Helpers ----------------
def prepare_batch(batch_prompts):
    messages = [
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": p}]
        for p in batch_prompts
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", padding=True).to(draft_model.device)

def forward_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

def sample_from_probs(probs):
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1)

# ---------------- Target-only batched ----------------
@torch.no_grad()
def run_target_only_batch(batch_prompts):
    inputs = prepare_batch(batch_prompts)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    orig_len = input_ids.shape[1]

    start = time.perf_counter()

    while input_ids.shape[1] - orig_len < max_new_tokens:
        logits = forward_logits(target_model, input_ids, attention_mask)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_tokens = sample_from_probs(probs)

        input_ids = torch.cat([input_ids, next_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
            dim=1
        )

    elapsed = time.perf_counter() - start
    tokens = (input_ids.shape[1] - orig_len) * input_ids.size(0)
    return elapsed, tokens

# ---------------- Speculative batched ----------------
@torch.no_grad()
def run_speculative_batch(batch_prompts, K):
    inputs = prepare_batch(batch_prompts)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    orig_len = input_ids.shape[1]

    start = time.perf_counter()

    while input_ids.shape[1] - orig_len < max_new_tokens:
        # Draft K tokens
        drafted = []
        tmp_ids = input_ids
        tmp_mask = attention_mask

        for _ in range(K):
            logits = forward_logits(draft_model, tmp_ids, tmp_mask)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            tok = sample_from_probs(probs)
            drafted.append(tok)
            tmp_ids = torch.cat([tmp_ids, tok], dim=1)
            tmp_mask = torch.cat(
                [tmp_mask, torch.ones((tmp_mask.size(0), 1), device=tmp_mask.device)],
                dim=1
            )

        drafted_tokens = torch.cat(drafted, dim=1)

        # Verify in parallel
        ext_ids = torch.cat([input_ids, drafted_tokens], dim=1)
        ext_mask = torch.cat([attention_mask, torch.ones_like(drafted_tokens)], dim=1)

        draft_logits = forward_logits(draft_model, ext_ids, ext_mask)
        target_logits = forward_logits(target_model, ext_ids, ext_mask)

        accepted = 0
        residual_probs = None

        for i in range(drafted_tokens.size(1)):
            pos = input_ids.size(1) + i - 1
            dp = torch.softmax(draft_logits[:, pos, :], dim=-1)
            tp = torch.softmax(target_logits[:, pos, :], dim=-1)

            tok = drafted_tokens[:, i]
            alpha = torch.clamp(tp.gather(1, tok.unsqueeze(1)) / dp.gather(1, tok.unsqueeze(1)), max=1.0)

            accept_mask = torch.rand_like(alpha) <= alpha
            if accept_mask.all():
                accepted += 1
            else:
                residual_probs = torch.clamp(tp - dp, min=0.0)
                break

        # Append accepted
        if accepted > 0:
            input_ids = torch.cat([input_ids, drafted_tokens[:, :accepted]], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), accepted), device=attention_mask.device)],
                dim=1
            )

        # Handle next token
        if residual_probs is not None:
            next_tok = sample_from_probs(residual_probs)
        else:
            pos = input_ids.shape[1] - 1
            probs = torch.softmax(target_logits[:, pos, :] / temperature, dim=-1)
            next_tok = sample_from_probs(probs)

        input_ids = torch.cat([input_ids, next_tok], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)],
            dim=1
        )

    elapsed = time.perf_counter() - start
    tokens = (input_ids.shape[1] - orig_len) * input_ids.size(0)
    return elapsed, tokens

# ---------------- Benchmark loops ----------------
speedup_vs_K = []
for K in Ks:
    print(f"Running K={K}")
    t_spec, n_spec = run_speculative_batch(prompts[:batch_sizes[0]], K)
    t_tgt, n_tgt = run_target_only_batch(prompts[:batch_sizes[0]])
    speedup_vs_K.append((n_spec / t_spec) / (n_tgt / t_tgt))

speedup_vs_B = []
for B in batch_sizes:
    print(f"Running B={B}")
    t_spec, n_spec = run_speculative_batch(prompts[:B], Ks[-1])
    t_tgt, n_tgt = run_target_only_batch(prompts[:B])
    speedup_vs_B.append((n_spec / t_spec) / (n_tgt / t_tgt))

# ---------------- Plots ----------------
plt.figure()
plt.plot(Ks, speedup_vs_K, marker='o')
plt.xlabel("K (draft tokens)")
plt.ylabel("Speedup vs target-only")
plt.title("Speculative decoding speedup vs K")
plt.savefig('speedup_vs_K.png', dpi=300, bbox_inches='tight')


plt.figure()
plt.plot(batch_sizes, speedup_vs_B, marker='o')
plt.xlabel("Batch size")
plt.ylabel("Speedup vs target-only")
plt.title("Speculative decoding speedup vs batch size")
plt.savefig('speedup_vs_batch.png', dpi=300, bbox_inches='tight')
