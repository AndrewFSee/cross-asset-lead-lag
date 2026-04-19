"""DeltaLag-style cross-attention model for dynamic lead-lag discovery.

A minimal port of the ACM AIF 2025 "DeltaLag" idea: for each candidate leader
asset and each candidate lag ℓ ∈ {1, …, L_max}, learn an attention score that
says "how useful is leader i at lag ℓ for predicting the target at t+1?"
The top-k (leader, lag) pairs are retained as a sparse lag graph, and their
attention-weighted signals are combined linearly to produce a forecast.

Two design choices keep this faithful to the paper while staying small:

* **Per-leader lag softmax.** Attention is normalised *within* each leader
  across the lag axis, so the model emits one learned lag per leader rather
  than one global lag. This is the "variable-lag" behaviour.
* **Pairwise rank-logistic loss.** For cross-sectional selection what matters
  is ordering, not level. We optimise a pairwise margin over random pairs
  of samples in a batch: the model wins if sign(y_i − y_j) matches
  sign(ŷ_i − ŷ_j). This is much more stable for thin-edge return prediction
  than MSE.

The module intentionally has a small surface: fit / predict / leader_lags.
`signals.backtest.WalkForwardBacktest` can call `.fit(train_slice)` on each
window then `.predict(next_row)` for the forward step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_lag_matrix(R: np.ndarray, max_lag: int) -> np.ndarray:
    """Stack lagged returns into (T-max_lag, max_lag, n_assets).

    Row t (0-indexed from `max_lag`) contains [r[t-1], …, r[t-max_lag]]
    along axis 0 (the lag axis) for every asset.
    """
    T, N = R.shape
    out = np.zeros((T - max_lag, max_lag, N), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        out[:, lag - 1, :] = R[max_lag - lag : T - lag, :]
    return out


class DeltaLagModel(nn.Module):
    """Cross-attention model jointly selecting (leader, lag) pairs.

    Inputs to ``forward`` are lag-stacked tensors of shape
    ``(batch, max_lag, n_leaders)`` and target-history vectors of shape
    ``(batch, max_lag)``. Output is a scalar per batch row — the predicted
    next-step return of the target.

    The attention over lags is computed as softmax over a learned
    ``(n_leaders, max_lag)`` logit matrix multiplied by the instantaneous
    (leader, lag) return. An optional top-k mask sparsifies the resulting
    ``(n_leaders, max_lag)`` attention grid so only the k strongest pairs
    are active.
    """

    def __init__(
        self,
        n_leaders: int,
        max_lag: int,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_leaders = n_leaders
        self.max_lag = max_lag
        self.top_k = top_k
        self.temperature = temperature

        self.lag_logits = nn.Parameter(torch.zeros(n_leaders, max_lag))
        self.leader_scale = nn.Parameter(torch.zeros(n_leaders))
        self.bias = nn.Parameter(torch.zeros(1))

    def attention(self) -> torch.Tensor:
        """Return the (n_leaders, max_lag) softmax attention grid."""
        logits = self.lag_logits / self.temperature
        if self.top_k is not None and self.top_k < self.max_lag:
            kth, _ = torch.topk(logits, self.top_k, dim=-1)
            thresh = kth[:, -1:].detach()
            logits = torch.where(logits >= thresh, logits, torch.full_like(logits, -1e9))
        return F.softmax(logits, dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, max_lag, n_leaders) → (batch,)."""
        att = self.attention()
        leader_signal = (X * att.T.unsqueeze(0)).sum(dim=1)
        scale = F.softplus(self.leader_scale)
        return (leader_signal * scale).sum(dim=-1) + self.bias

    def leader_lags(self) -> np.ndarray:
        """Argmax lag for each leader (1-indexed)."""
        with torch.no_grad():
            return (self.lag_logits.argmax(dim=-1).cpu().numpy() + 1).astype(int)


def _rank_logistic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pairwise rank-logistic loss over all B*(B-1) ordered pairs in a batch.

    For every pair (i, j), penalise when the sign of (pred_i - pred_j)
    disagrees with sign(target_i - target_j). Equivalent to optimising
    Kendall's τ via its smooth logistic surrogate.
    """
    dp = pred.unsqueeze(0) - pred.unsqueeze(1)
    dy = target.unsqueeze(0) - target.unsqueeze(1)
    sign = torch.sign(dy)
    mask = sign != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.softplus(-sign[mask] * dp[mask]).mean()


def fit_delta_lag(
    leader_returns: np.ndarray,
    target_returns: np.ndarray,
    max_lag: int = 10,
    top_k: Optional[int] = None,
    n_epochs: int = 200,
    learning_rate: float = 5e-2,
    l1_penalty: float = 1e-3,
    loss: str = "rank",
    random_state: int = 0,
    device: Optional[str] = None,
) -> Tuple[DeltaLagModel, Dict[str, float]]:
    """Fit a DeltaLagModel on a single training window.

    Args:
        leader_returns: Shape (T, n_leaders). Each column is one candidate
            leader asset's per-bar return stream.
        target_returns: Shape (T,). Per-bar return of the target asset.
        max_lag: Largest candidate lag in bars.
        top_k: If set, only keep top-k lags per leader in the attention grid.
        n_epochs: Adam steps on the full training batch.
        learning_rate: Adam LR.
        l1_penalty: L1 on ``leader_scale`` for sparsity across leaders.
        loss: "rank" (pairwise logistic) or "mse".
        random_state: Torch seed for reproducibility.
        device: "cpu" / "cuda"; auto-detected if None.

    Returns:
        (trained_model, training_metrics_dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    R = np.asarray(leader_returns, dtype=np.float32)
    y = np.asarray(target_returns, dtype=np.float32)
    if R.shape[0] != y.shape[0]:
        raise ValueError("leader_returns and target_returns must share length")
    T, N = R.shape
    if T <= max_lag + 10:
        raise ValueError(f"Need T > max_lag+10; got T={T}, max_lag={max_lag}")

    X = _stack_lag_matrix(R, max_lag)
    y_aligned = y[max_lag:]

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y_aligned).to(device)

    model = DeltaLagModel(n_leaders=N, max_lag=max_lag, top_k=top_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses: List[float] = []
    for _ in range(n_epochs):
        opt.zero_grad()
        pred = model(X_t)
        if loss == "rank":
            core = _rank_logistic_loss(pred, y_t)
        else:
            core = F.mse_loss(pred, y_t)
        reg = l1_penalty * F.softplus(model.leader_scale).sum()
        total = core + reg
        total.backward()
        opt.step()
        losses.append(float(core.item()))

    with torch.no_grad():
        pred_final = model(X_t).cpu().numpy()
    ic = float(np.corrcoef(pred_final, y_aligned)[0, 1]) if pred_final.std() > 1e-12 else 0.0

    return model, {
        "final_loss": losses[-1],
        "train_ic": ic,
        "n_train": int(len(y_aligned)),
    }


def predict_delta_lag(
    model: DeltaLagModel,
    leader_returns: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """Apply a fitted DeltaLagModel to a new window of leader returns.

    Args:
        model: Trained ``DeltaLagModel``.
        leader_returns: Shape (T, n_leaders) with T > model.max_lag. The
            last row's prediction is the forecast for *the next* bar.

    Returns:
        Array of length T - max_lag of predicted returns.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    R = np.asarray(leader_returns, dtype=np.float32)
    if R.shape[1] != model.n_leaders:
        raise ValueError("n_leaders mismatch between model and input")
    X = _stack_lag_matrix(R, model.max_lag)
    X_t = torch.from_numpy(X).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def delta_lag_leader_graph(
    model: DeltaLagModel,
    leader_names: List[str],
    target_name: str,
    min_weight: float = 1e-3,
) -> List[Dict]:
    """Export the fitted attention grid as a tidy lead→target edge list.

    Each row is one (leader, lag) edge with its attention weight and the
    leader's overall scale (softplus of ``leader_scale``). Edges below
    ``min_weight`` (on the attention * scale product) are dropped.
    """
    att = model.attention().detach().cpu().numpy()
    scale = F.softplus(model.leader_scale).detach().cpu().numpy()
    edges: List[Dict] = []
    for i, name in enumerate(leader_names):
        for lag in range(model.max_lag):
            w = float(att[i, lag] * scale[i])
            if w >= min_weight:
                edges.append({
                    "source": name,
                    "target": target_name,
                    "lag": int(lag + 1),
                    "attention": float(att[i, lag]),
                    "scale": float(scale[i]),
                    "weight": w,
                })
    return edges
