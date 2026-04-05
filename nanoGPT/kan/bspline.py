"""B-spline grid and basis function evaluation using PyTorch.

Provides uniform cubic B-spline grids with Cox-de Boor evaluation,
Greville abscissae for identity initialization, and dynamic grid expansion.
All operations are vectorized over batch dimensions using torch tensors.
"""

import math
import torch
import torch.nn.functional as F


class BSplineGrid:
    """Uniform B-spline grid with vectorized basis evaluation.

    The grid covers [grid_min, grid_max] with uniform knot spacing.
    Supports dynamic expansion when inputs fall outside the grid range.
    """

    def __init__(self, order: int, num_basis: int, grid_min: float, grid_max: float):
        self.order = order
        self.num_basis = num_basis
        self.grid_min = grid_min
        self.grid_max = grid_max

        num_interior = max(num_basis - order, 1)
        self.step = (grid_max - grid_min) / num_interior

        num_knots = num_basis + order + 1
        self.knots = torch.tensor(
            [grid_min + (i - order) * self.step for i in range(num_knots)],
            dtype=torch.float32,
        )

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions at points x.

        Args:
            x: tensor of shape (...,) — any batch shape

        Returns:
            Tensor of shape (..., num_basis) — basis function values
        """
        flat = x.reshape(-1)
        t = self.knots
        k = self.order + 1  # degree + 1
        num_intervals = len(t) - 1

        # Degree-0 basis: piecewise constant
        # basis[i] = 1 if t[i] <= x < t[i+1], with right endpoint included for last interval
        t_lo = t[:num_intervals]  # (num_intervals,)
        t_hi = t[1 : num_intervals + 1]  # (num_intervals,)

        x_exp = flat.unsqueeze(-1)  # (N, 1)
        basis = ((x_exp >= t_lo) & (x_exp < t_hi)).float()  # (N, num_intervals)

        # Include right endpoint for the last interval
        basis[:, -1] += (flat == t[-1]).float()

        # Cox-de Boor recursion for degrees 1..order
        for d in range(1, k):
            n_new = num_intervals - d
            t_left = t[:n_new]
            t_right = t[d : d + n_new]
            denom1 = t_right - t_left
            safe1 = denom1.clamp(min=1e-10)
            left = (x_exp - t_left) / safe1 * basis[:, :n_new]
            left = left * (denom1 > 0).float()

            t_left2 = t[1 : n_new + 1]
            t_right2 = t[d + 1 : d + 1 + n_new]
            denom2 = t_right2 - t_left2
            safe2 = denom2.clamp(min=1e-10)
            right = (t_right2 - x_exp) / safe2 * basis[:, 1 : n_new + 1]
            right = right * (denom2 > 0).float()

            basis = left + right

        # Trim or pad to num_basis
        if basis.shape[-1] > self.num_basis:
            basis = basis[:, : self.num_basis]
        elif basis.shape[-1] < self.num_basis:
            basis = F.pad(basis, (0, self.num_basis - basis.shape[-1]))

        return basis.reshape(*x.shape, self.num_basis)

    def greville_abscissae(self) -> torch.Tensor:
        """Compute Greville abscissae — coefficients that make KAN(x) = x.

        The Greville abscissa for basis function i is the average of k-1
        consecutive knots starting at position i+1. When used as B-spline
        coefficients, the resulting curve is the identity function.
        """
        k = self.order + 1
        coeffs = torch.zeros(self.num_basis)
        for i in range(self.num_basis):
            indices = [i + j for j in range(1, k) if i + j < len(self.knots)]
            if indices:
                coeffs[i] = self.knots[indices].mean()
        return coeffs

    def expand(self, new_min: float, new_max: float) -> tuple["BSplineGrid", int]:
        """Expand grid to cover [new_min, new_max], preserving step size.

        Returns (new_grid, left_offset) where left_offset is the number
        of new basis functions added on the left. Old basis i maps to
        new basis (i + left_offset).
        """
        left_steps = 0
        if new_min < self.grid_min:
            left_steps = math.ceil((self.grid_min - new_min) / self.step)
        right_steps = 0
        if new_max > self.grid_max:
            right_steps = math.ceil((new_max - self.grid_max) / self.step)

        if left_steps == 0 and right_steps == 0:
            return self, 0

        expanded_min = self.grid_min - left_steps * self.step
        expanded_max = self.grid_max + right_steps * self.step
        new_num_basis = self.num_basis + left_steps + right_steps

        new_grid = BSplineGrid(self.order, new_num_basis, expanded_min, expanded_max)
        new_grid.step = self.step  # Force same step to avoid floating-point drift
        return new_grid, left_steps
