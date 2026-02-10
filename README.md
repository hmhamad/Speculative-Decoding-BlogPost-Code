# Speculative Decoding from Scratch

Companion code for the blog post [Speculative Decoding: Fast LLM Inference Without Quality Loss](https://www.hassanhamad.com/blog/speculative_decoding/speculative-decoding.html).

## Code

The code is organized as a progression of programs, each building on the previous:

| File | Description |
|------|-------------|
| `p0.py` | Baseline using `model.generate()` |
| `p1.py` | Manual token-by-token decoding loop |
| `p2.py` | Generalized K-token generation |
| `p3.py` | Draft model + verification logic |
| `p4.py` | Full speculative decoding loop |
| `p5.py` | Benchmarks (draft vs target vs speculative) |

Each file is self-contained and can be run independently.

## Setup & Run

Using [uv](https://docs.astral.sh/uv/) (or any Python package manager of your choice):

```bash
uv sync
uv run p0.py   # or any other file
```

Note: The code uses `device = "mps"` for Apple Silicon. Change to `"cuda"` for NVIDIA GPUs.

## Citation

Please cite this work as:

```
Hamad, Hassan, "Speculative Decoding: Fast LLM Inference Without Quality Loss", hassanhamad.com, Nov 2025.
```

BibTeX:

```bibtex
@article{hamad2025speculative,
  author = {Hassan Hamad},
  title = {Speculative Decoding: Fast LLM Inference Without Quality Loss},
  journal = {hassanhamad.com},
  year = {2025},
  note = {https://hassanhamad.com/blog/speculative_decoding/},
}
```
