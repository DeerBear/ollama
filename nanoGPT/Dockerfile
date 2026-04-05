# Geometric KAN Attention A/B Demo for nanoGPT
#
# The A/B test IS the git history:
#   - Branch "baseline": Karpathy's unmodified nanoGPT
#   - Branch "kan-attention": same code + KAN files only
#   - `git diff baseline..kan-attention` = the COMPLETE change set
#
# Both branches train on the same data, same config, same seed.
# The ONLY difference is the attention mechanism.
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
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    numpy tiktoken datasets transformers

# ── Step 3: Create the "baseline" branch ─────────────────────────────
# Tag the exact unmodified state of nanoGPT. This is the A in A/B.
RUN cd /app/nanoGPT && \
    git checkout -b baseline && \
    git tag -a v0-baseline -m "Unmodified nanoGPT from Karpathy"

# ── Step 4: Prepare training data ────────────────────────────────────
# Gutenberg, Dammit by Allison Parrish — real multilingual literature.
# https://github.com/aparrish/gutenberg-dammit
COPY prepare_gutenberg.py /app/nanoGPT/data/gutenberg_dammit/prepare.py
RUN cd /app/nanoGPT && GUTENBERG_MAX_TEXTS=1000 python data/gutenberg_dammit/prepare.py

# Shakespeare too — for quick sanity checks
RUN cd /app/nanoGPT && python data/shakespeare_char/prepare.py

# Data prep is NOT a code change — don't commit it to baseline.
# The .bin files live in data/ which is .gitignored by nanoGPT.

# ── Step 5: Train the baseline model ─────────────────────────────────
# Same config for both branches. Same seed. Same data.
RUN cd /app/nanoGPT && python train.py \
    config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --max_iters=2000 \
    --eval_interval=500 \
    --dataset=gutenberg_dammit \
    --out_dir=out-gutenberg \
    --seed_offset=42

# Checkpoint lives in out-gutenberg/ — shared by both branches.

# ── Step 6: Verify baseline is pristine ──────────────────────────────
RUN cd /app/nanoGPT && \
    echo "=== Baseline branch: nanoGPT source is UNMODIFIED ===" && \
    git diff --stat HEAD && \
    git status

# ── Step 7: Create the "kan-attention" branch ────────────────────────
# Branch from baseline. The diff between branches IS the experiment.
RUN cd /app/nanoGPT && git checkout -b kan-attention

# ── Step 8: Add the ONLY custom code ─────────────────────────────────
COPY kan/ /app/nanoGPT/kan/
COPY kan_integrate.py /app/nanoGPT/kan_integrate.py
COPY demo.py /app/nanoGPT/demo.py

# ── Step 9: Commit the KAN code to the kan-attention branch ──────────
RUN cd /app/nanoGPT && \
    git add kan/ kan_integrate.py demo.py && \
    git -c user.name="KAN Demo" -c user.email="kan@demo" \
    commit -m "Add Geometric KAN attention (inference-time learning)

Files added:
  kan/              — B-spline KAN implementation
  kan_integrate.py  — runtime monkey-patch for attention layers
  demo.py           — idempotent A/B comparison script

nanoGPT source code is UNTOUCHED.
The diff between 'baseline' and 'kan-attention' branches
is the COMPLETE set of changes. Nothing else."

# ── Step 10: Show the proof ──────────────────────────────────────────
# This prints during build so anyone reading the build log can see it.
RUN cd /app/nanoGPT && \
    echo "" && \
    echo "════════════════════════════════════════════════════════" && \
    echo "  THE A/B TEST IS THE GIT DIFF" && \
    echo "════════════════════════════════════════════════════════" && \
    echo "" && \
    echo "Branches:" && \
    git branch -v && \
    echo "" && \
    echo "Diff (baseline vs kan-attention):" && \
    git diff --stat baseline..kan-attention && \
    echo "" && \
    echo "To see full diff: git diff baseline..kan-attention" && \
    echo "════════════════════════════════════════════════════════"

ENTRYPOINT []
CMD ["python", "/app/nanoGPT/demo.py"]
