"""Geometric mean normalization and clip-redistribute for KAN coefficients.

These are utility functions, not called during initialization or Phase 1
training (which would destroy the identity property). Available for use
in Phase 2 evolution or manual coefficient management.
"""

import torch


def geometric_mean(weights: torch.Tensor) -> torch.Tensor:
    """Geometric mean of absolute values: exp(mean(log(|w|)))."""
    abs_w = weights.abs().clamp(min=1e-10)
    return torch.exp(torch.mean(torch.log(abs_w)))


def geometric_mean_normalize(weights: torch.Tensor) -> torch.Tensor:
    """Scale weights so geometric mean of |weights| = 1."""
    gm = geometric_mean(weights)
    if gm < 1e-10:
        return weights
    return weights / gm


def clip_and_redistribute(weights: torch.Tensor) -> torch.Tensor:
    """Adaptive clipping with proportional excess redistribution.

    Threshold = mean(|w|) + 2*std(|w|)

    Weights exceeding the threshold are clipped, and the excess energy
    is redistributed to all coefficients proportional to their share
    of the total absolute weight. Preserves signs.
    """
    abs_w = weights.abs()
    threshold = abs_w.mean() + 2 * abs_w.std()
    if threshold < 1e-10:
        return weights

    signs = weights.sign()
    excess_mask = abs_w > threshold
    excess = (abs_w - threshold).clamp(min=0)
    total_excess = excess.sum()

    clipped = torch.where(excess_mask, signs * threshold, weights)

    if total_excess < 1e-10:
        return clipped

    clipped_abs = clipped.abs()
    abs_sum = clipped_abs.sum()
    if abs_sum < 1e-10:
        return clipped

    shares = clipped_abs / abs_sum
    return clipped + signs * shares * total_excess


def normalize_and_redistribute(weights: torch.Tensor) -> torch.Tensor:
    """Full pipeline: geo-mean normalize, clip-redistribute, re-normalize."""
    w = geometric_mean_normalize(weights)
    w = clip_and_redistribute(w)
    w = geometric_mean_normalize(w)
    return w
