"""KAN integration shim for nanoGPT.

Runtime monkey-patch that replaces softmax with Geometric KAN attention.
nanoGPT's source code is NOT modified — this patches the forward method
of each CausalSelfAttention module after the model is loaded.

This file + kan/ directory are the ONLY custom code. Everything else
is Karpathy's unmodified nanoGPT, cloned fresh from GitHub.
"""

import types
import logging
import torch
import torch.nn.functional as F

from kan.shadow import ShadowTrainer

logger = logging.getLogger(__name__)


def patch_attention_with_kan(model, trainer: ShadowTrainer):
    """Replace softmax with KAN in all CausalSelfAttention layers.

    For each attention layer, replaces the forward method with one that:
    1. Computes Q, K, V projections (unchanged)
    2. Computes attention scores QK^T / sqrt(d_k) (unchanged)
    3. Applies causal mask (unchanged)
    4. Routes through KAN instead of softmax (THE ONLY CHANGE)
    5. Applies attention to V (unchanged)
    """
    layer_count = 0

    for block_idx, block in enumerate(model.transformer.h):
        attn = block.attn
        layer_key = f"layer_{block_idx}"

        def make_kan_forward(key):
            def kan_forward(self_attn, x):
                B, T, C = x.size()

                # Q, K, V projections — UNCHANGED from nanoGPT
                q, k, v = self_attn.c_attn(x).split(self_attn.n_embd, dim=2)
                n_head = self_attn.n_head
                head_dim = C // n_head

                k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                q = q.view(B, T, n_head, head_dim).transpose(1, 2)
                v = v.view(B, T, n_head, head_dim).transpose(1, 2)

                # Attention scores — UNCHANGED from nanoGPT
                att = (q @ k.transpose(-2, -1)) * (1.0 / head_dim ** 0.5)

                # Causal mask — UNCHANGED from nanoGPT
                if hasattr(self_attn, "bias"):
                    att = att.masked_fill(
                        self_attn.bias[:, :, :T, :T] == 0, float("-inf")
                    )

                # ════════════════════════════════════════════════
                # THIS IS THE ONLY CHANGE: KAN replaces softmax
                # ════════════════════════════════════════════════
                softmax_out = F.softmax(att, dim=-1)
                softmax_out = self_attn.attn_dropout(softmax_out)

                if not trainer.is_converged(key):
                    # Phase 1: train KAN in shadow, use softmax output
                    trainer.train_step(key, att.detach(), softmax_out.detach())
                    att_weights = softmax_out
                else:
                    # Post-convergence: use KAN output directly
                    kan = trainer.get_kan(key)
                    with torch.no_grad():
                        att_weights = kan(att)

                    # Phase 2: self-evolution (optional)
                    state = trainer.layers.get(key)
                    if state and state.phase2_active:
                        trainer.phase2_step(key, att.detach())

                # Apply attention to V — UNCHANGED from nanoGPT
                y = att_weights @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self_attn.resid_dropout(self_attn.c_proj(y))
                return y

            return kan_forward

        attn.forward = types.MethodType(make_kan_forward(layer_key), attn)
        layer_count += 1
        logger.info(f"Patched attention layer {block_idx} -> {layer_key}")

    logger.info(f"KAN attention installed on {layer_count} layers")
    return trainer
