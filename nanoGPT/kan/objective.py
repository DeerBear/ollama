"""Pluggable objective functions for KAN training.

Each objective defines three functions:
1. Phase 1 loss (supervised: match a target)
2. Phase 2 objective (self-supervised: evolve beyond target)
3. Drift metric (safety: don't diverge too far from graduation)
"""

import torch
import torch.nn.functional as F


class Objective:
    """Base class for KAN training objectives."""

    def phase1_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def phase2_objective(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def drift_metric(self, reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AttentionObjective(Objective):
    """Objective for softmax attention replacement.

    Phase 1: MSE between softmax output and KAN output.
    Phase 2: Attention sharpness (negative entropy) — sharper = better.
    Drift: KL divergence from graduation checkpoint.
    """

    def phase1_loss(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, target)

    def phase2_objective(self, output: torch.Tensor) -> torch.Tensor:
        """Negative entropy: sum(p * log(p)), averaged over rows.

        Higher = sharper attention. Maximized via gradient ascent.
        """
        p = output.clamp(min=1e-10)
        entropy = (p * p.log()).sum(dim=-1)  # negative entropy per row
        return entropy.mean()

    def drift_metric(self, reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """KL(reference || current), averaged over rows."""
        ref = reference.clamp(min=1e-10)
        cur = current.clamp(min=1e-10)
        kl = (ref * (ref.log() - cur.log())).sum(dim=-1)
        return kl.mean()
