"""Shadow trainer: online KAN learning at inference time.

Phase 1: KAN runs alongside softmax, learns to match it via MSE.
         Uses real PyTorch autograd — no finite-difference hacks.
Phase 2: After convergence, KAN self-evolves by maximizing attention
         sharpness (negative entropy) with a KL divergence safety rail.

Each transformer layer gets its own KAN + optimizer + convergence state.
"""

import logging
import torch
import torch.optim as optim

from kan.layer import KANLayer
from kan.objective import Objective, AttentionObjective

logger = logging.getLogger(__name__)


class LayerState:
    """Per-layer training state."""

    def __init__(self, kan: KANLayer, lr: float, betas: tuple, eps: float):
        self.kan = kan
        self.optimizer = optim.Adam(kan.parameters(), lr=lr, betas=betas, eps=eps)
        self.step_count = 0
        self.ema_loss = 0.0
        self.convergence_count = 0
        self.converged = False

        # Phase 2
        self.phase2_active = False
        self.phase2_optimizer = None
        self.graduation_snapshot = None
        self.graduation_slope = 1.0
        self.effective_scale = 1.0
        self.phase2_steps = 0
        self.ema_sharpness = 0.0

        # Plateau detection for head spawning
        self.best_loss = float("inf")
        self.plateau_count = 0
        self.last_head_spawn = 0


class ShadowTrainer:
    """Manages online KAN training across all transformer layers.

    Usage:
        trainer = ShadowTrainer()

        # During each forward pass:
        softmax_out = F.softmax(logits, dim=-1)
        kan_out = trainer.train_step("layer_0", logits, softmax_out)
    """

    def __init__(
        self,
        order: int = 3,
        num_basis: int = 8,
        grid_min: float = -5.0,
        grid_max: float = 5.0,
        lr: float = 1e-3,
        phase2_lr: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        convergence_threshold: float = 1e-4,
        convergence_window: int = 50,
        train_every_n: int = 1,
        phase2_enabled: bool = True,
        phase2_every_n: int = 10,
        phase2_max_drift: float = 0.1,
        max_heads: int = 3,
        plateau_window: int = 200,
        plateau_improvement: float = 0.05,
        objective: Objective = None,
    ):
        self.order = order
        self.num_basis = num_basis
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.lr = lr
        self.phase2_lr = phase2_lr
        self.betas = betas
        self.eps = eps
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.train_every_n = train_every_n
        self.phase2_enabled = phase2_enabled
        self.phase2_every_n = phase2_every_n
        self.phase2_max_drift = phase2_max_drift
        self.max_heads = max_heads
        self.plateau_window = plateau_window
        self.plateau_improvement = plateau_improvement
        self.objective = objective or AttentionObjective()

        self.layers: dict[str, LayerState] = {}

    def _get_or_create(self, key: str) -> LayerState:
        if key not in self.layers:
            kan = KANLayer(self.order, self.num_basis, self.grid_min, self.grid_max)
            self.layers[key] = LayerState(kan, self.lr, self.betas, self.eps)
        return self.layers[key]

    def is_converged(self, key: str) -> bool:
        state = self.layers.get(key)
        return state.converged if state else False

    def is_fully_converged(self) -> bool:
        if not self.layers:
            return False
        return all(s.converged for s in self.layers.values())

    @torch.no_grad()
    def train_step(
        self, key: str, logits: torch.Tensor, softmax_out: torch.Tensor
    ) -> float:
        """One Phase 1 training step: make KAN match softmax.

        Args:
            key: layer identifier (e.g. "layer_0")
            logits: pre-softmax attention logits (batch, n_head, seq_q, seq_k)
            softmax_out: ground truth softmax output (same shape)

        Returns:
            Current EMA loss.
        """
        state = self._get_or_create(key)
        state.step_count += 1

        if state.converged:
            return state.ema_loss

        if state.step_count % self.train_every_n != 0:
            return state.ema_loss

        # Enable gradients for this training step
        with torch.enable_grad():
            for p in state.kan.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            kan_out = state.kan(logits)
            loss = self.objective.phase1_loss(softmax_out, kan_out)
            loss.backward()
            state.optimizer.step()

        loss_val = loss.item()

        # EMA tracking
        if state.ema_loss == 0:
            state.ema_loss = loss_val
            state.best_loss = loss_val
        else:
            state.ema_loss = 0.99 * state.ema_loss + 0.01 * loss_val

        # Plateau detection for head spawning
        if state.ema_loss < state.best_loss * (1.0 - self.plateau_improvement):
            state.best_loss = state.ema_loss
            state.plateau_count = 0
        else:
            state.plateau_count += 1

        if (
            state.plateau_count >= self.plateau_window
            and state.kan.num_heads < self.max_heads
            and state.step_count - state.last_head_spawn > self.plateau_window
        ):
            num_heads = state.kan.add_head()
            # Rebuild optimizer to include new head's parameters
            state.optimizer = optim.Adam(
                state.kan.parameters(), lr=self.lr, betas=self.betas, eps=self.eps
            )
            state.plateau_count = 0
            state.best_loss = state.ema_loss
            state.last_head_spawn = state.step_count
            logger.info(
                f"KAN spawned new head: layer={key} heads={num_heads} "
                f"ema_loss={state.ema_loss:.6f} step={state.step_count}"
            )

        # Convergence check
        if state.ema_loss < self.convergence_threshold:
            state.convergence_count += 1
            if state.convergence_count >= self.convergence_window:
                state.converged = True
                state.graduation_slope = self._raw_slope(state.kan)
                state.effective_scale = 1.0

                if self.phase2_enabled:
                    state.phase2_active = True
                    state.graduation_snapshot = state.kan.snapshot()
                    state.phase2_optimizer = optim.Adam(
                        state.kan.parameters(),
                        lr=self.phase2_lr,
                        betas=self.betas,
                        eps=self.eps,
                    )
                logger.info(
                    f"KAN converged: layer={key} ema_loss={state.ema_loss:.6f} "
                    f"steps={state.step_count}"
                )
        else:
            state.convergence_count = 0

        if state.step_count % 100 == 0:
            logger.debug(
                f"KAN progress: layer={key} step={state.step_count} "
                f"ema_loss={state.ema_loss:.6f}"
            )

        return state.ema_loss

    @torch.no_grad()
    def phase2_step(self, key: str, logits: torch.Tensor) -> tuple[float, bool]:
        """One Phase 2 step: self-evolve past softmax.

        Args:
            key: layer identifier
            logits: current attention logits

        Returns:
            (sharpness, drifted) — if drifted, KAN was reverted to graduation.
        """
        state = self.layers.get(key)
        if not state or not state.phase2_active:
            return 0.0, False

        state.phase2_steps += 1
        if state.phase2_steps % self.phase2_every_n != 0:
            return state.ema_sharpness, False

        # Gradient ascent on Phase 2 objective
        with torch.enable_grad():
            for p in state.kan.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            kan_out = state.kan(logits)
            objective = self.objective.phase2_objective(kan_out)
            # Negate because we want to MAXIMIZE, but optimizer does gradient descent
            (-objective).backward()
            state.phase2_optimizer.step()

        # Check drift from graduation
        with torch.no_grad():
            current_out = state.kan(logits)
            grad_out = state.graduation_snapshot(logits)
            drift = self.objective.drift_metric(grad_out, current_out).item()

        if drift > self.phase2_max_drift:
            logger.warning(
                f"KAN Phase 2 drift exceeded, reverting: layer={key} "
                f"drift={drift:.4f} max={self.phase2_max_drift}"
            )
            # Revert to graduation weights
            for p, g in zip(state.kan.parameters(), state.graduation_snapshot.parameters()):
                p.data.copy_(g.data)
            state.phase2_optimizer = optim.Adam(
                state.kan.parameters(),
                lr=self.phase2_lr,
                betas=self.betas,
                eps=self.eps,
            )
            state.effective_scale = 1.0
            return state.ema_sharpness, True

        # Update tracking
        sharpness = self.objective.phase2_objective(current_out).item()
        if state.ema_sharpness == 0:
            state.ema_sharpness = sharpness
        else:
            state.ema_sharpness = 0.95 * state.ema_sharpness + 0.05 * sharpness

        if state.graduation_slope > 0:
            state.effective_scale = self._raw_slope(state.kan) / state.graduation_slope

        if state.phase2_steps % 100 == 0:
            logger.debug(
                f"KAN Phase 2: layer={key} step={state.phase2_steps} "
                f"sharpness={state.ema_sharpness:.4f} drift={drift:.4f} "
                f"scale={state.effective_scale:.4f}"
            )

        return state.ema_sharpness, False

    def get_kan(self, key: str) -> KANLayer:
        """Get the KAN layer for a given key (creates if needed)."""
        return self._get_or_create(key).kan

    def get_effective_scale(self, key: str) -> float:
        state = self.layers.get(key)
        if state and state.effective_scale > 0:
            return state.effective_scale
        return 1.0

    def stats(self) -> dict:
        converged = sum(1 for s in self.layers.values() if s.converged)
        total_steps = sum(s.step_count for s in self.layers.values())
        return {
            "total_layers": len(self.layers),
            "converged": converged,
            "total_steps": total_steps,
            "fully_converged": converged == len(self.layers) and len(self.layers) > 0,
        }

    @staticmethod
    def _raw_slope(kan: KANLayer) -> float:
        """Estimate linear slope of KAN transform via least-squares."""
        points = torch.tensor([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        with torch.no_grad():
            y = kan.evaluate_raw(points)
        sum_xy = (points * y).sum().item()
        sum_xx = (points * points).sum().item()
        if sum_xx == 0:
            return 1.0
        slope = sum_xy / sum_xx
        return slope if slope > 0 else 1.0
