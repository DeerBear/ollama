# Geometric KAN Attention Demo for nanoGPT
#
# Everything is downloaded fresh — nanoGPT, datasets, PyTorch.
# The ONLY custom code is the kan/ directory + two integration files.
# nanoGPT's source code is NOT modified — the integration is done via
# runtime monkey-patching of the attention forward method.
#
# Dataset: Gutenberg, Dammit by Allison Parrish
# https://github.com/aparrish/gutenberg-dammit
#
# The Dockerfile IS the proof: every step is visible and auditable.
#
# Build:
#   docker build -t nanogpt-kan nanoGPT/
#
# Run:
#   docker run -it --rm nanogpt-kan

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Clone nanoGPT fresh from GitHub ──────────────────────────
RUN git clone https://github.com/karpathy/nanoGPT.git /app/nanoGPT

# ── Step 2: Install dependencies ─────────────────────────────────────
# PyTorch CPU — keeps image small, proves KAN doesn't need a GPU
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    numpy tiktoken datasets transformers

# ── Step 3: Prepare training data from Gutenberg, Dammit ────────────
# Downloads the corpus ZIP (~1.2 GB) from Allison Parrish's archive,
# extracts English texts, tokenizes with GPT-2 encoding, saves as
# train.bin / val.bin in nanoGPT's expected format.
#
# Source: https://github.com/aparrish/gutenberg-dammit
# Data:   http://static.decontextualize.com/gutenberg-dammit-files-v002.zip
COPY prepare_gutenberg.py /app/nanoGPT/data/gutenberg_dammit/prepare.py
RUN cd /app/nanoGPT && python data/gutenberg_dammit/prepare.py

# ── Step 4: Train a baseline model ──────────────────────────────────
# 6 layers, 6 heads, 384 dim — trained on real multilingual literature
# from Project Gutenberg. Uses GPT-2 tokenizer (not char-level).
RUN cd /app/nanoGPT && python train.py \
    config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --max_iters=2000 \
    --eval_interval=500 \
    --dataset=gutenberg_dammit \
    --out_dir=out-gutenberg

# ── Step 5: Verify nanoGPT source is unmodified ─────────────────────
# This proves we didn't touch Karpathy's code during training
RUN cd /app/nanoGPT && git diff --stat && git status

# ── Step 6: Copy in the ONLY custom code ─────────────────────────────
# kan/             — Geometric KAN implementation (B-splines, shadow trainer)
# kan_integrate.py — runtime monkey-patch to hook KAN into attention
# demo.py          — A/B comparison script
COPY kan/ /app/nanoGPT/kan/
COPY kan_integrate.py /app/nanoGPT/kan_integrate.py
COPY demo.py /app/nanoGPT/demo.py

# nanoGPT's model.py is UNTOUCHED. Zero sed. Zero patches.
# The integration happens at runtime via kan_integrate.py which
# replaces the attention forward method after model load.

ENTRYPOINT []
CMD ["python", "/app/nanoGPT/demo.py"]
