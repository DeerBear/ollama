"""Prepare Gutenberg, Dammit corpus for nanoGPT training.

Downloads the corpus ZIP from Allison Parrish's gutenberg-dammit project,
extracts English texts, tokenizes with tiktoken (GPT-2 encoding), and
saves as train.bin / val.bin in nanoGPT's expected format.

Source: https://github.com/aparrish/gutenberg-dammit
"""

import os
import json
import zipfile
import urllib.request
import numpy as np
import tiktoken

CORPUS_URL = "http://static.decontextualize.com/gutenberg-dammit-files-v002.zip"
CORPUS_ZIP = "/app/gutenberg-dammit/gutenberg-dammit-files-v002.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SPLIT = 0.9

# Max texts to include (0 = all). Set lower for faster builds.
MAX_TEXTS = int(os.environ.get("GUTENBERG_MAX_TEXTS", "0"))


def download_corpus():
    """Download the corpus ZIP if not already present."""
    if os.path.exists(CORPUS_ZIP):
        print(f"Corpus already exists at {CORPUS_ZIP}")
        return

    print(f"Downloading corpus from {CORPUS_URL}...")
    print("(This is ~1.2 GB, may take a few minutes)")
    urllib.request.urlretrieve(CORPUS_URL, CORPUS_ZIP)
    print("Download complete.")


def extract_texts():
    """Extract English texts from the corpus ZIP."""
    print("Reading metadata and extracting English texts...")
    texts = []
    total_chars = 0

    with zipfile.ZipFile(CORPUS_ZIP, "r") as zf:
        # Load metadata
        meta_raw = zf.read("gutenberg-metadata.json").decode("utf-8")
        metadata = json.loads(meta_raw)

        count = 0
        for entry in metadata:
            # Filter for English texts
            languages = entry.get("Language", [])
            if "en" not in languages:
                continue

            gd_path = entry.get("gd-path")
            if not gd_path:
                continue

            try:
                text = zf.read(gd_path).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue

            if len(text.strip()) < 100:
                continue

            texts.append(text)
            total_chars += len(text)
            count += 1

            if count % 1000 == 0:
                print(f"  Extracted {count} texts ({total_chars / 1e6:.1f}M chars)...")

            if MAX_TEXTS > 0 and count >= MAX_TEXTS:
                print(f"  Stopping at {MAX_TEXTS} texts (GUTENBERG_MAX_TEXTS)")
                break

    print(f"Extracted {len(texts)} English texts ({total_chars / 1e6:.1f}M chars)")
    return texts


def tokenize_and_save(texts):
    """Tokenize with GPT-2 encoding and save as train/val .bin files."""
    print("Tokenizing with tiktoken (GPT-2 encoding)...")
    enc = tiktoken.get_encoding("gpt2")

    # Concatenate all texts with double newline separator
    all_text = "\n\n".join(texts)
    tokens = enc.encode_ordinary(all_text)
    tokens = np.array(tokens, dtype=np.uint16)

    print(f"Total tokens: {len(tokens):,}")

    # Split into train/val
    split_idx = int(len(tokens) * TRAIN_SPLIT)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val:   {len(val_tokens):,} tokens")

    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Saved: {train_path} ({os.path.getsize(train_path) / 1e6:.1f}MB)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path) / 1e6:.1f}MB)")


if __name__ == "__main__":
    download_corpus()
    texts = extract_texts()
    tokenize_and_save(texts)
    print("Done! Ready for nanoGPT training.")
