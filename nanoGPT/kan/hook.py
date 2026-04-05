"""PyTorch hooks to install KAN attention into any transformer model.

Works with nanoGPT and any model that computes attention as:
    attn_weights = softmax(QK^T / sqrt(d_k))

The hook intercepts the attention scores before softmax and routes them
through the KAN shadow trainer. During Phase 1 the model still uses
softmax output; after convergence the KAN output is used directly.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from kan.shadow import ShadowTrainer

logger = logging.getLogger(__name__)


class KANAttentionHook:
    """Hook that replaces softmax with KAN in an attention module.

    Attaches to an attention layer and intercepts the forward pass.
    The hook computes both softmax and KAN outputs:
    - Before convergence: uses softmax, trains KAN in shadow
    - After convergence: uses KAN output directly
    """

    def __init__(self, layer_name: str, trainer: ShadowTrainer):
        self.layer_name = layer_name
        self.trainer = trainer
        self.handle = None

    def hook_fn(self, module, args, output):
        """Post-forward hook on the attention module.

        This is model-specific. For nanoGPT's CausalSelfAttention,
        we need to intercept the attention computation. Since PyTorch
        hooks see inputs/outputs of the whole module, we use a wrapper
        approach instead (see wrap_attention).
        """
        pass  # Wrapper approach is cleaner; see wrap_attention below


def wrap_attention(attn_module: nn.Module, layer_name: str, trainer: ShadowTrainer):
    """Wrap a nanoGPT CausalSelfAttention to use KAN.

    Replaces the module's forward method with one that:
    1. Computes Q, K, V projections normally
    2. Computes attention scores (QK^T / sqrt(d_k))
    3. Applies causal mask
    4. Routes through KAN instead of softmax (after convergence)
    5. Applies attention to V

    This is the cleanest integration point — we replace exactly
    the softmax call and nothing else.
    """
    original_forward = attn_module.forward

    def kan_forward(x):
        B, T, C = x.size()
        # Use the module's projection layers
        # nanoGPT packs Q, K, V into one linear projection
        qkv = attn_module.c_attn(x)
        q, k, v = qkv.split(attn_module.n_embd, dim=2)

        n_head = attn_module.n_head
        head_dim = C // n_head

        # Reshape for multi-head attention
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # Attention scores
        scale = 1.0 / (head_dim ** 0.5)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)

        # Causal mask
        att = att.masked_fill(
            attn_module.bias[:, :, :T, :T] == 0, float("-inf")
        )

        # === This is where KAN replaces softmax ===
        softmax_out = F.softmax(att, dim=-1)
        softmax_out = attn_module.attn_dropout(softmax_out)

        if not trainer.is_converged(layer_name):
            # Phase 1: train KAN in shadow, use softmax output
            trainer.train_step(layer_name, att.detach(), softmax_out.detach())
            attn_weights = softmax_out
        else:
            # Post-convergence: use KAN output
            kan = trainer.get_kan(layer_name)
            with torch.no_grad():
                attn_weights = kan(att)

            # Phase 2: self-evolution
            if trainer.layers[layer_name].phase2_active:
                trainer.phase2_step(layer_name, att.detach())

        # Apply attention to values
        y = attn_weights @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = attn_module.resid_dropout(attn_module.c_proj(y))
        return y

    attn_module.forward = kan_forward
    return attn_module


def install_kan_attention(
    model: nn.Module,
    trainer: ShadowTrainer = None,
    layer_prefix: str = "transformer.h",
    attn_attr: str = "attn",
) -> ShadowTrainer:
    """Install KAN attention on all attention layers in a model.

    Args:
        model: the transformer model (e.g. nanoGPT's GPT)
        trainer: existing ShadowTrainer, or None to create one
        layer_prefix: prefix for finding transformer blocks
        attn_attr: attribute name for attention module within each block

    Returns:
        The ShadowTrainer managing all KAN layers.

    Example:
        trainer = install_kan_attention(model)
        # Now every forward pass trains the KAN automatically
    """
    if trainer is None:
        trainer = ShadowTrainer()

    layer_count = 0
    for name, module in model.named_modules():
        # Match attention modules in transformer blocks
        if name.endswith(f".{attn_attr}") and layer_prefix in name:
            layer_key = f"layer_{layer_count}"
            wrap_attention(module, layer_key, trainer)
            logger.info(f"Installed KAN attention: {name} -> {layer_key}")
            layer_count += 1

    if layer_count == 0:
        # Fallback: try to find any module with 'attn' in the name
        for name, module in model.named_modules():
            if "attn" in name.lower() and hasattr(module, "c_attn"):
                layer_key = f"layer_{layer_count}"
                wrap_attention(module, layer_key, trainer)
                logger.info(f"Installed KAN attention (fallback): {name} -> {layer_key}")
                layer_count += 1

    logger.info(f"KAN attention installed on {layer_count} layers")
    return trainer
