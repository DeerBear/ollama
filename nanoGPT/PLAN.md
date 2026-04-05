# Geometric KAN Attention for nanoGPT

## Goal

Replace softmax in a pretrained nanoGPT model with a **Geometric KAN** (Kolmogorov-Arnold Network) at **inference time** — no retraining required.

The KAN learns to match softmax first (Phase 1), then evolves beyond it (Phase 2), all while the model is running.

## What is Geometric KAN Attention?

Standard attention: `attn = softmax(QK^T / sqrt(d_k))`

KAN attention: `attn = normalize(exp(KAN(QK^T / sqrt(d_k))))`

Where `KAN(x)` is a learnable B-spline transform that:
1. **Starts as identity** (so `exp(x - max) / sum(exp)` = softmax exactly)
2. **Shadow-trains** alongside softmax to match it perfectly (Phase 1)
3. **Self-evolves** past softmax to sharpen attention (Phase 2)

Key innovations:
- **B-spline basis functions** (cubic, local support, smooth) instead of a fixed exponential
- **Greville abscissae initialization** for perfect identity at step 0
- **Dynamic grid expansion** so any logit range works (critical for distilled models)
- **Multi-head cooperative KAN** — heads combine additively in log-space
- **Geometric mean normalization** with excess redistribution for coefficient management

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Transformer Attention Layer                     │
│                                                  │
│  logits = Q @ K^T / sqrt(d_k) + mask            │
│                                                  │
│  ┌──────────────┐    ┌───────────────────────┐   │
│  │   softmax    │    │   Geometric KAN       │   │
│  │  (standard)  │    │                       │   │
│  │              │    │  B-spline basis eval   │   │
│  │  target for  │───>│  + multi-head coeffs   │   │
│  │  Phase 1     │    │  + exp-normalize       │   │
│  │              │    │                        │   │
│  └──────┬───────┘    └───────────┬───────────┘   │
│         │                        │                │
│         │    Phase 1: match      │                │
│         │    Phase 2: evolve     │                │
│         v                        v                │
│  attn_weights (used for V @ attn)                │
└─────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `kan/bspline.py` | B-spline grid, knot vectors, Cox-de Boor evaluation, Greville abscissae, dynamic grid expansion |
| `kan/coefficients.py` | Coefficient management, geometric mean normalization, clip-and-redistribute |
| `kan/layer.py` | Multi-head KAN layer: forward pass, expansion, snapshot |
| `kan/shadow.py` | Shadow trainer: Phase 1 (match softmax), Phase 2 (evolve), Adam optimizer, convergence tracking, plateau-triggered head spawning |
| `kan/objective.py` | Pluggable objective interface + AttentionObjective (MSE / sharpness / KL divergence) |
| `kan/hook.py` | PyTorch hook to intercept attention and apply KAN |
| `kan/__init__.py` | Package exports |

## Training Phases

### Phase 1: Shadow Training (match softmax)
- KAN runs alongside softmax on every forward pass
- Loss = MSE(softmax_output, kan_output)
- Adam optimizer with real gradients (PyTorch autograd)
- Converges when EMA loss < 1e-4 for 50 consecutive steps
- At convergence: KAN output ≈ softmax output (graduation)

### Phase 2: Self-Evolution (exceed softmax)
- After graduation, KAN optimizes attention sharpness (negative entropy)
- Gradient ascent on: `sharpness = mean(sum(p * log(p)))` per row
- Safety rail: KL divergence from graduation checkpoint < 0.1
- If drift exceeds threshold, revert to graduation weights

### Dynamic Features
- **Grid expansion**: if logits fall outside B-spline range, grid grows automatically
- **Head spawning**: if loss plateaus for 200 steps, a new cooperative head is added (max 3)
- **Effective scale tracking**: ratio of current B-spline slope to graduation slope

## How to Use

```python
import torch
from kan.hook import install_kan_attention

# Load your pretrained nanoGPT model
model = load_model("path/to/checkpoint.pt")

# Install KAN attention hooks on all attention layers
hooks = install_kan_attention(model)

# Run inference — KAN trains automatically in the background
for batch in dataloader:
    output = model(batch)
    # KAN is learning with every forward pass

# Check convergence
for name, hook in hooks.items():
    print(f"{name}: converged={hook.trainer.is_converged(name)}")
```

## Design Philosophy

This is a **clean-room reimplementation**, not a port. The Go version has
workarounds for infrastructure limitations (finite-difference gradients because
GGML has no autograd, manual mutexes for concurrency, manual array copies for
tensor ops). None of that belongs in Python.

This version uses:
- **PyTorch autograd** for real gradients (no finite-difference hacks)
- **torch.optim.Adam** (no hand-rolled optimizer)
- **Vectorized tensor ops** (no scalar loops)
- **nn.Module / nn.Parameter** (standard PyTorch patterns)
- **Forward method wrapping** (no GGML graph surgery)

The math is identical. The implementation is what PyTorch was designed for.
Nobody can argue with `F.mse_loss`, `torch.optim.Adam`, or `F.softmax`.
