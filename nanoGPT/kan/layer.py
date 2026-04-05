"""Multi-head Geometric KAN layer.

Replaces softmax: takes pre-softmax logits, applies learned B-spline transforms,
produces attention weights that sum to 1 per query position.

Multiple heads cooperate additively in log-space, each specializing on a
different part of the error surface. New heads are spawned as zero (no-op)
so they don't disrupt existing heads.
"""

import torch
import torch.nn as nn

from kan.bspline import BSplineGrid


class KANLayer(nn.Module):
    """Multi-head B-spline KAN that replaces softmax normalization.

    Forward: logits -> B-spline transform -> exp-normalize -> attention weights

    The B-spline transform is initialized to identity (Greville abscissae),
    so the full pipeline starts as exactly softmax.
    """

    def __init__(self, order: int = 3, num_basis: int = 8,
                 grid_min: float = -5.0, grid_max: float = 5.0):
        super().__init__()
        self.grid = BSplineGrid(order, num_basis, grid_min, grid_max)

        # Head 0: identity initialization (Greville abscissae)
        init_coeffs = self.grid.greville_abscissae()
        self.heads = nn.ParameterList([nn.Parameter(init_coeffs)])

    @property
    def num_heads(self) -> int:
        return len(self.heads)

    def add_head(self):
        """Spawn a new cooperative head initialized to zero (no-op)."""
        zeros = torch.zeros(self.grid.num_basis)
        self.heads.append(nn.Parameter(zeros))
        return self.num_heads

    def expand_if_needed(self, logits: torch.Tensor):
        """Expand grid if logits fall outside current range.

        Head 0 gets identity coefficients for new basis functions.
        Other heads get zeros (remain no-op in expanded region).
        """
        lo = logits.min().item()
        hi = logits.max().item()

        if lo >= self.grid.grid_min and hi <= self.grid.grid_max:
            return

        new_grid, left_offset = self.grid.expand(lo, hi)
        greville = new_grid.greville_abscissae()

        new_heads = []
        for h, head in enumerate(self.heads):
            new_w = torch.zeros(new_grid.num_basis)
            if h == 0:
                # Primary head: fill with Greville (identity), overlay trained
                new_w.copy_(greville)
            # Copy old coefficients into shifted positions
            new_w[left_offset : left_offset + len(head.data)] = head.data
            new_heads.append(nn.Parameter(new_w))

        self.heads = nn.ParameterList(new_heads)
        self.grid = new_grid

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply multi-head KAN + exp-normalize to attention logits.

        Args:
            logits: (batch, n_head, seq_q, seq_k) pre-softmax attention scores

        Returns:
            Attention weights of same shape, each row sums to 1.
        """
        self.expand_if_needed(logits)

        # Evaluate B-spline basis at every logit value
        # basis shape: (*logits.shape, num_basis)
        basis = self.grid.evaluate(logits)

        # Sum across all heads: f(x) = sum_h sum_i c_hi * B_i(x)
        raw = torch.zeros_like(logits)
        for head in self.heads:
            # head: (num_basis,), basis: (..., num_basis)
            raw = raw + (basis * head).sum(dim=-1)

        # Row-wise numerically stable exp-normalize (same as softmax)
        row_max = raw.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(raw - row_max)
        attn = exp_scores / exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        return attn

    def evaluate_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Raw B-spline transform without exp-normalize.

        Useful for debugging and slope estimation.
        """
        basis = self.grid.evaluate(x)
        raw = torch.zeros_like(x)
        for head in self.heads:
            raw = raw + (basis * head).sum(dim=-1)
        return raw

    def snapshot(self) -> "KANLayer":
        """Create a detached copy for gradient estimation or checkpointing."""
        snap = KANLayer.__new__(KANLayer)
        nn.Module.__init__(snap)
        snap.grid = self.grid
        snap.heads = nn.ParameterList([
            nn.Parameter(h.data.clone()) for h in self.heads
        ])
        return snap
