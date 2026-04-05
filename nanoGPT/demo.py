#!/usr/bin/env python3
"""Geometric KAN Attention A/B Demo for nanoGPT.

This script runs a fair, reproducible comparison:
1. BASELINE: Generate text with standard softmax attention
2. KAN: Generate text with KAN attention (shadow-trained at inference time)
3. COMPARISON: Side-by-side output with perplexity measurements

Everything is deterministic with fixed seeds. The model was trained during
Docker image build — no training happens here except KAN shadow training.
"""

import os
import sys
import logging
import time

import torch
import torch.nn.functional as F
import numpy as np

# nanoGPT is at /app/nanoGPT
sys.path.insert(0, "/app/nanoGPT")
os.chdir("/app/nanoGPT")

from model import GPT, GPTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kan-demo")

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_DIR = "out-shakespeare-char"
DEVICE = "cpu"
SEED = 42
NUM_TOKENS = 200  # tokens to generate per prompt
KAN_WARMUP_PROMPTS = 5  # prompts to warm up KAN before measuring
TEMPERATURE = 0.8
TOP_K = 40

TEST_PROMPTS = [
    "To be, or not to be, that is the question",
    "All that glitters is not",
    "The quality of mercy is not",
    "Now is the winter of our",
    "Friends, Romans, countrymen, lend me your",
    "Out, out, brief candle! Life's but a walking",
    "If music be the food of love",
    "We are such stuff as dreams are made",
]

WARMUP_PROMPTS = [
    "Once upon a time there was a king who",
    "In the beginning of the world, when all was",
    "The sun set slowly over the mountains and",
    "A wise man once said that the truth of",
    "There lived in a small village a young",
]

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load the pretrained nanoGPT checkpoint."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"No checkpoint found at {ckpt_path}")
        logger.error("The model should have been trained during Docker build.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = checkpoint["model_args"]
    conf = GPTConfig(**model_args)
    model = GPT(conf)

    # Strip the "module." prefix if saved with DDP
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(DEVICE)
    return model, checkpoint


def load_tokenizer():
    """Load the character-level tokenizer from Shakespeare data."""
    import pickle
    meta_path = os.path.join("data/shakespeare_char", "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])
    return encode, decode


@torch.no_grad()
def generate(model, encode, decode, prompt, num_tokens=NUM_TOKENS,
             temperature=TEMPERATURE, top_k=TOP_K, seed=SEED):
    """Generate text from a prompt. Returns the generated continuation."""
    torch.manual_seed(seed)
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(num_tokens):
        # Crop to block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    # Return only the generated part
    generated = idx[0, len(encode(prompt)):].tolist()
    return decode(generated)


@torch.no_grad()
def compute_perplexity(model, encode, text):
    """Compute perplexity on a text string."""
    tokens = encode(text)
    if len(tokens) < 2:
        return float("inf")

    x = torch.tensor(tokens[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = torch.tensor(tokens[1:], dtype=torch.long, device=DEVICE).unsqueeze(0)

    # Process in chunks if longer than block_size
    block_size = model.config.block_size
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, x.size(1), block_size):
        x_chunk = x[:, i:i + block_size]
        y_chunk = y[:, i:i + block_size]
        logits, _ = model(x_chunk)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_chunk.view(-1))
        total_loss += loss.item() * y_chunk.size(1)
        total_tokens += y_chunk.size(1)

    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)


# ═══════════════════════════════════════════════════════════════════════
# Colors (ANSI)
# ═══════════════════════════════════════════════════════════════════════
BOLD = "\033[1m"
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"


def header(text):
    print(f"\n{BOLD}{CYAN}{'═' * 60}{NC}")
    print(f"{BOLD}{CYAN}  {text}{NC}")
    print(f"{BOLD}{CYAN}{'═' * 60}{NC}\n")


def info(text):
    print(f"{GREEN}[INFO]{NC} {text}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    header("Geometric KAN Attention A/B Demo")
    info("Model: nanoGPT (Shakespeare char-level, 6L/6H/384D)")
    info("Source: github.com/karpathy/nanoGPT (unmodified)")
    info("Custom code: kan/ directory only")
    print()

    # Load model and tokenizer
    info("Loading pretrained model...")
    model, checkpoint = load_model()
    encode, decode = load_tokenizer()
    info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # ─── Step 1: Baseline (standard softmax) ─────────────────────────
    header("Step 1: Baseline (Standard Softmax)")

    baseline_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{BOLD}{CYAN}Test {i+1}: {prompt}...{NC}")
        result = generate(model, encode, decode, prompt)
        print(f"  {result[:200]}")
        baseline_results.append(result)
        print()

    # Baseline perplexity on generated text
    info("Computing baseline perplexity...")
    baseline_ppls = []
    for prompt, result in zip(TEST_PROMPTS, baseline_results):
        ppl = compute_perplexity(model, encode, prompt + result)
        baseline_ppls.append(ppl)
    info(f"Baseline mean perplexity: {np.mean(baseline_ppls):.2f}")
    print()

    # ─── Step 2: Install KAN attention ───────────────────────────────
    header("Step 2: Installing KAN Attention")

    from kan.shadow import ShadowTrainer
    from kan_integrate import patch_attention_with_kan

    trainer = ShadowTrainer()
    patch_attention_with_kan(model, trainer)

    info(f"KAN installed on {len(list(model.transformer.h))} attention layers")
    print()

    # ─── Step 3: Warm up the KAN ─────────────────────────────────────
    header("Step 3: KAN Training (Warm-Up)")
    info(f"Running {KAN_WARMUP_PROMPTS} warm-up prompts to train KAN...")
    print()

    for i, prompt in enumerate(WARMUP_PROMPTS[:KAN_WARMUP_PROMPTS]):
        info(f"Warm-up {i+1}/{KAN_WARMUP_PROMPTS}: {prompt[:50]}...")
        _ = generate(model, encode, decode, prompt, num_tokens=300)

        stats = trainer.stats()
        converged = stats["converged"]
        total = stats["total_layers"]
        info(f"  Converged: {converged}/{total} layers")

    print()
    info(f"Warm-up complete. Final stats: {trainer.stats()}")
    print()

    # ─── Step 4: KAN inference ───────────────────────────────────────
    header("Step 4: KAN Attention Inference")

    kan_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{BOLD}{CYAN}Test {i+1}: {prompt}...{NC}")
        result = generate(model, encode, decode, prompt)
        print(f"  {result[:200]}")
        kan_results.append(result)
        print()

    # KAN perplexity
    info("Computing KAN perplexity...")
    kan_ppls = []
    for prompt, result in zip(TEST_PROMPTS, kan_results):
        ppl = compute_perplexity(model, encode, prompt + result)
        kan_ppls.append(ppl)
    info(f"KAN mean perplexity: {np.mean(kan_ppls):.2f}")
    print()

    # ─── Step 5: Side-by-side comparison ─────────────────────────────
    header("A/B Comparison")

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{BOLD}{CYAN}Test {i+1}: {prompt}...{NC}")
        print(f"  {RED}Baseline:{NC} {baseline_results[i][:150]}")
        print(f"  {GREEN}KAN:     {NC} {kan_results[i][:150]}")
        print(f"  {YELLOW}PPL:     {NC} baseline={baseline_ppls[i]:.2f}  kan={kan_ppls[i]:.2f}")
        print()

    # ─── Final scoreboard ────────────────────────────────────────────
    header("Final Scoreboard")

    baseline_mean = np.mean(baseline_ppls)
    kan_mean = np.mean(kan_ppls)

    print(f"  {RED}Baseline mean perplexity: {baseline_mean:.2f}{NC}")
    print(f"  {GREEN}KAN mean perplexity:      {kan_mean:.2f}{NC}")
    print()

    kan_wins = sum(1 for b, k in zip(baseline_ppls, kan_ppls) if k < b)
    baseline_wins = sum(1 for b, k in zip(baseline_ppls, kan_ppls) if b < k)
    ties = len(baseline_ppls) - kan_wins - baseline_wins

    print(f"  {GREEN}KAN wins:      {kan_wins}/{len(TEST_PROMPTS)} (lower perplexity){NC}")
    print(f"  {RED}Baseline wins: {baseline_wins}/{len(TEST_PROMPTS)}{NC}")
    print(f"  {YELLOW}Ties:          {ties}/{len(TEST_PROMPTS)}{NC}")
    print()

    if kan_mean < baseline_mean:
        improvement = (1 - kan_mean / baseline_mean) * 100
        print(f"  {BOLD}{GREEN}KAN improves perplexity by {improvement:.1f}%{NC}")
    elif baseline_mean < kan_mean:
        regression = (kan_mean / baseline_mean - 1) * 100
        print(f"  {BOLD}{RED}KAN regresses perplexity by {regression:.1f}%{NC}")
    else:
        print(f"  {BOLD}{YELLOW}Dead tie{NC}")

    print()

    # Final convergence stats
    header("KAN Training Stats")
    stats = trainer.stats()
    for k, v in stats.items():
        info(f"  {k}: {v}")

    for key, state in trainer.layers.items():
        info(f"  {key}: converged={state.converged} steps={state.step_count} "
             f"ema_loss={state.ema_loss:.6f} heads={state.kan.num_heads}")

    print()
    info("Done. All code is auditable in the Dockerfile and kan/ directory.")


if __name__ == "__main__":
    main()
