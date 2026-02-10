import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
target_model_name = "Qwen/Qwen3-4B-Instruct-2507"

device = "mps"
temperature = 0.8
max_new_tokens = 128
K = 5
num_runs = 1   # average timings

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
# Prompt setup
# ------------------------------------------------------------
prompt = "List the top 10 countries by population in decreasing order."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

def init_inputs():
    inputs = tokenizer(text, return_tensors="pt").to(draft_model.device)
    return inputs.input_ids, inputs.attention_mask

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def sample_from_probs(probs):
    probs = probs / probs.sum()
    return torch.multinomial(probs, 1)

@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

# ------------------------------------------------------------
# Scenario 1: draft-only decoding
# ------------------------------------------------------------
@torch.no_grad()
def run_draft_only():
    input_ids, attention_mask = init_inputs()
    orig_len = input_ids.shape[1]

    start = time.perf_counter()

    while input_ids.shape[1] - orig_len < max_new_tokens:
        logits = forward_logits(draft_model, input_ids, attention_mask)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        if next_token.item() == eos_id:
            break

    elapsed = time.perf_counter() - start
    tokens = input_ids.shape[1] - orig_len
    return elapsed, tokens

# ------------------------------------------------------------
# Scenario 2: target-only decoding
# ------------------------------------------------------------
@torch.no_grad()
def run_target_only():
    input_ids, attention_mask = init_inputs()
    orig_len = input_ids.shape[1]

    start = time.perf_counter()

    while input_ids.shape[1] - orig_len < max_new_tokens:
        logits = forward_logits(target_model, input_ids, attention_mask)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        if next_token.item() == eos_id:
            break

    elapsed = time.perf_counter() - start
    tokens = input_ids.shape[1] - orig_len
    return elapsed, tokens

# ------------------------------------------------------------
# Scenario 3: speculative decoding
# ------------------------------------------------------------
@torch.no_grad()
def run_speculative():
    input_ids, attention_mask = init_inputs()
    orig_len = input_ids.shape[1]

    drafted_total = 0
    accepted_total = 0
    full_accepts = 0
    iterations = 0

    start = time.perf_counter()

    while input_ids.shape[1] - orig_len < max_new_tokens:
        iterations += 1

        # -------- Draft K tokens --------
        drafted = []
        tmp_ids = input_ids
        tmp_mask = attention_mask

        for _ in range(K):
            logits = forward_logits(draft_model, tmp_ids, tmp_mask)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)

            drafted.append(tok)
            tmp_ids = torch.cat([tmp_ids, tok], dim=1)
            tmp_mask = torch.cat(
                [tmp_mask, torch.ones((1, 1), device=tmp_mask.device)],
                dim=1
            )

            if tok.item() == eos_id:
                break

        drafted_tokens = torch.cat(drafted, dim=1)
        drafted_total += drafted_tokens.shape[1]

        # -------- Verify in parallel --------
        extended_ids = torch.cat([input_ids, drafted_tokens], dim=1)
        extended_mask = torch.cat(
            [attention_mask, torch.ones_like(drafted_tokens)],
            dim=1
        )

        draft_logits = forward_logits(draft_model, extended_ids, extended_mask)
        target_logits = forward_logits(target_model, extended_ids, extended_mask)

        accepted = 0
        residual_probs = None

        for i in range(drafted_tokens.shape[1]):
            pos = input_ids.shape[1] + i - 1
            token_id = drafted_tokens[0, i]

            dp = torch.softmax(draft_logits[0, pos], dim=-1)
            tp = torch.softmax(target_logits[0, pos], dim=-1)

            alpha = torch.clamp(tp[token_id] / dp[token_id], max=1.0)

            if torch.rand(()) <= alpha:
                accepted += 1
                if token_id.item() == eos_id:
                    break
            else:
                residual_probs = torch.clamp(tp - dp, min=0.0)
                break

        accepted_total += accepted

        # -------- Append accepted --------
        for i in range(accepted):
            tok = drafted_tokens[:, i:i+1]
            input_ids = torch.cat([input_ids, tok], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
                dim=1
            )

        # -------- Handle rejection / full accept --------
        if residual_probs is not None:
            next_token = sample_from_probs(residual_probs).view(1, 1)
        else:
            full_accepts += 1
            pos = input_ids.shape[1] - 1
            probs = torch.softmax(target_logits[0, pos] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).view(1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device)],
            dim=1
        )

        if next_token.item() == eos_id:
            break

    elapsed = time.perf_counter() - start
    tokens = input_ids.shape[1] - orig_len

    stats = {
        "elapsed": elapsed,
        "tokens": tokens,
        "accept_rate": accepted_total / max(drafted_total, 1),
        "avg_accept_per_iter": accepted_total / max(iterations, 1),
        "full_accept_frac": full_accepts / max(iterations, 1),
    }

    return stats

# ------------------------------------------------------------
# Run benchmarks
# ------------------------------------------------------------
def avg_runs(fn):
    times, toks = [], []
    for _ in range(num_runs):
        t, n = fn()
        times.append(t)
        toks.append(n)
    return sum(times)/num_runs, sum(toks)/num_runs

draft_t, draft_n = avg_runs(run_draft_only)
target_t, target_n = avg_runs(run_target_only)

spec_stats = [run_speculative() for _ in range(num_runs)]

spec_time = sum(s["elapsed"] for s in spec_stats) / num_runs
spec_tokens = sum(s["tokens"] for s in spec_stats) / num_runs
spec_accept = sum(s["accept_rate"] for s in spec_stats) / num_runs
spec_full = sum(s["full_accept_frac"] for s in spec_stats) / num_runs

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
print("\n=== Benchmark Results ===\n")

print(f"Draft model:")
print(f"  time: {draft_t:.3f}s")
print(f"  tokens: {draft_n:.1f}")
print(f"  tok/s: {draft_n / draft_t:.1f}")

print(f"\nTarget model:")
print(f"  time: {target_t:.3f}s")
print(f"  tokens: {target_n:.1f}")
print(f"  tok/s: {target_n / target_t:.1f}")

print(f"\nSpeculative decoding:")
print(f"  time: {spec_time:.3f}s")
print(f"  tokens: {spec_tokens:.1f}")
print(f"  tok/s: {spec_tokens / spec_time:.1f}")
print(f"  acceptance rate: {spec_accept:.3f}")
print(f"  full accept fraction: {spec_full:.3f}")

print(f"\nSpeedup vs target-only: {(spec_tokens/spec_time)/(target_n/target_t):.2f}×")

