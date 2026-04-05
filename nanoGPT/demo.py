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
import tiktoken

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
CHECKPOINT_DIR = "out-gutenberg"
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

# Fixed held-out passages for idempotent perplexity measurement.
# These never change between baseline and KAN — the ONLY variable
# is the attention mechanism. Same model, same text, same tokens.
EVAL_PASSAGES = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
    "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair.",
    "Happy families are all alike; every unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys' house. The wife had discovered that the husband was carrying on an intrigue with a French girl, who had been a governess in their family.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show. To begin my life with the beginning of my life, I record that I was born on a Friday, at twelve o'clock at night.",
    "I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe; nor am I one of your Hollywood movie ectoplasms. I am a man of substance, of flesh and bone, fiber and liquids, and I might even be said to possess a mind.",
    "The cold passed reluctantly from the earth, and the retiring fogs revealed an army stretched out on the hills, resting. As the landscape changed from brown to green, the army awakened, and began to tremble with eagerness at the noise of rumors.",
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
    """Load the GPT-2 tokenizer (matches the Gutenberg training data)."""
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special=set())
    decode = lambda l: enc.decode(l)
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
    info("Model: nanoGPT (Gutenberg, 6L/6H/384D)")
    info("Source: github.com/karpathy/nanoGPT (unmodified)")
    info("Dataset: Gutenberg, Dammit by Allison Parrish")
    info("Custom code: kan/ directory only")
    print()

    # Lock down all randomness
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model and tokenizer
    info("Loading pretrained model...")
    model, checkpoint = load_model()
    encode, decode = load_tokenizer()
    info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # ─── Step 1: Baseline perplexity on FIXED passages ───────────────
    header("Step 1: Baseline Perplexity (Standard Softmax)")
    info("Measuring perplexity on 8 fixed literary passages.")
    info("These passages are IDENTICAL for baseline and KAN.")
    info("The ONLY variable is the attention mechanism.")
    print()

    baseline_ppls = []
    for i, passage in enumerate(EVAL_PASSAGES):
        ppl = compute_perplexity(model, encode, passage)
        baseline_ppls.append(ppl)
        print(f"  {BOLD}Passage {i+1}:{NC} {passage[:60]}...")
        print(f"  {RED}Baseline PPL: {ppl:.4f}{NC}")
        print()

    baseline_mean = np.mean(baseline_ppls)
    info(f"Baseline mean perplexity: {baseline_mean:.4f}")
    print()

    # ─── Step 2: Baseline generation (for visual comparison) ─────────
    header("Step 2: Baseline Generation")

    baseline_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        torch.manual_seed(SEED + i)  # Per-prompt seed for reproducibility
        print(f"{BOLD}{CYAN}Prompt {i+1}: {prompt}...{NC}")
        result = generate(model, encode, decode, prompt, seed=SEED + i)
        print(f"  {result[:200]}")
        baseline_results.append(result)
        print()

    # ─── Step 3: Install KAN attention ───────────────────────────────
    header("Step 3: Installing KAN Attention")

    from kan.shadow import ShadowTrainer
    from kan_integrate import patch_attention_with_kan

    trainer = ShadowTrainer()
    patch_attention_with_kan(model, trainer)

    info(f"KAN installed on {len(list(model.transformer.h))} attention layers")
    print()

    # ─── Step 4: Warm up the KAN ─────────────────────────────────────
    header("Step 4: KAN Training (Warm-Up)")
    info(f"Running {KAN_WARMUP_PROMPTS} warm-up prompts to shadow-train KAN...")
    info("KAN learns to match softmax during these forward passes.")
    print()

    for i, prompt in enumerate(WARMUP_PROMPTS[:KAN_WARMUP_PROMPTS]):
        info(f"Warm-up {i+1}/{KAN_WARMUP_PROMPTS}: {prompt[:50]}...")
        _ = generate(model, encode, decode, prompt, num_tokens=300, seed=SEED + 100 + i)

        stats = trainer.stats()
        converged = stats["converged"]
        total = stats["total_layers"]
        info(f"  Converged: {converged}/{total} layers")

    print()
    info(f"Warm-up complete. Final stats: {trainer.stats()}")
    print()

    # ─── Step 5: KAN perplexity on THE SAME fixed passages ──────────
    header("Step 5: KAN Perplexity (Same Passages, Same Model)")
    info("Measuring perplexity on the EXACT SAME 8 passages.")
    info("Same model weights. Same tokens. Only attention changed.")
    print()

    kan_ppls = []
    for i, passage in enumerate(EVAL_PASSAGES):
        ppl = compute_perplexity(model, encode, passage)
        kan_ppls.append(ppl)
        print(f"  {BOLD}Passage {i+1}:{NC} {passage[:60]}...")
        print(f"  {GREEN}KAN PPL:      {ppl:.4f}{NC}")
        print(f"  {RED}Baseline PPL: {baseline_ppls[i]:.4f}{NC}")
        delta = baseline_ppls[i] - ppl
        pct = (delta / baseline_ppls[i]) * 100 if baseline_ppls[i] > 0 else 0
        marker = GREEN + "BETTER" if delta > 0 else (RED + "WORSE" if delta < 0 else YELLOW + "SAME")
        print(f"  {marker} ({pct:+.4f}%){NC}")
        print()

    kan_mean = np.mean(kan_ppls)
    info(f"KAN mean perplexity: {kan_mean:.4f}")
    print()

    # ─── Step 6: KAN generation (visual comparison) ──────────────────
    header("Step 6: KAN Generation")

    kan_results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        torch.manual_seed(SEED + i)
        print(f"{BOLD}{CYAN}Prompt {i+1}: {prompt}...{NC}")
        result = generate(model, encode, decode, prompt, seed=SEED + i)
        print(f"  {result[:200]}")
        kan_results.append(result)
        print()

    # ─── Final scoreboard ────────────────────────────────────────────
    header("RESULTS: Idempotent Perplexity Comparison")
    info("Same model. Same passages. Same tokens. Only attention differs.")
    print()

    for i, passage in enumerate(EVAL_PASSAGES):
        delta = baseline_ppls[i] - kan_ppls[i]
        pct = (delta / baseline_ppls[i]) * 100 if baseline_ppls[i] > 0 else 0
        marker = GREEN if delta > 0 else (RED if delta < 0 else YELLOW)
        print(f"  Passage {i+1}: baseline={baseline_ppls[i]:.4f}  "
              f"kan={kan_ppls[i]:.4f}  {marker}delta={pct:+.4f}%{NC}")
    print()

    print(f"  {RED}Baseline mean: {baseline_mean:.4f}{NC}")
    print(f"  {GREEN}KAN mean:      {kan_mean:.4f}{NC}")
    print()

    overall_delta = baseline_mean - kan_mean
    overall_pct = (overall_delta / baseline_mean) * 100 if baseline_mean > 0 else 0

    kan_wins = sum(1 for b, k in zip(baseline_ppls, kan_ppls) if k < b)
    baseline_wins = sum(1 for b, k in zip(baseline_ppls, kan_ppls) if b < k)
    ties = len(baseline_ppls) - kan_wins - baseline_wins

    print(f"  {GREEN}KAN wins:      {kan_wins}/{len(EVAL_PASSAGES)}{NC}")
    print(f"  {RED}Baseline wins: {baseline_wins}/{len(EVAL_PASSAGES)}{NC}")
    print(f"  {YELLOW}Ties:          {ties}/{len(EVAL_PASSAGES)}{NC}")
    print()

    if overall_pct > 0:
        print(f"  {BOLD}{GREEN}KAN improves perplexity by {overall_pct:.4f}%{NC}")
    elif overall_pct < 0:
        print(f"  {BOLD}{RED}KAN regresses perplexity by {-overall_pct:.4f}%{NC}")
    else:
        print(f"  {BOLD}{YELLOW}Dead tie{NC}")
    print()

    # ─── Generation side-by-side (qualitative) ───────────────────────
    header("Generation Comparison (Qualitative)")
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"{BOLD}{CYAN}Prompt: {prompt}...{NC}")
        print(f"  {RED}Baseline:{NC} {baseline_results[i][:150]}")
        print(f"  {GREEN}KAN:     {NC} {kan_results[i][:150]}")
        print()

    # ─── KAN training stats ──────────────────────────────────────────
    header("KAN Training Stats")
    stats = trainer.stats()
    for k, v in stats.items():
        info(f"  {k}: {v}")

    for key, state in trainer.layers.items():
        info(f"  {key}: converged={state.converged} steps={state.step_count} "
             f"ema_loss={state.ema_loss:.6f} heads={state.kan.num_heads}")

    print()
    info("This test is IDEMPOTENT: run it again, get the same numbers.")
    info("All code is auditable in the Dockerfile and kan/ directory.")


if __name__ == "__main__":
    main()
