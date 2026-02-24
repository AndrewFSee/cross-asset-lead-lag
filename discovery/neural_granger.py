"""Neural Granger Causality using component-wise LSTM (Tank et al. 2021)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ComponentLSTM(nn.Module):
    """Component-wise LSTM for neural Granger causality.

    Each input asset has a separate linear embedding. A shared 2-layer LSTM
    processes the concatenated embeddings to predict the next return of the
    target asset. Ablation of individual input embeddings reveals Granger
    causal influence.

    Args:
        n_inputs: Number of input assets (including the target).
        hidden_dim: LSTM hidden state dimension.
        n_layers: Number of LSTM layers.
        embed_dim: Embedding dimension per asset.
        ablate_idx: If provided, zero out this asset's embedding (ablation test).
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        embed_dim: int = 16,
        ablate_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.ablate_idx = ablate_idx

        # Separate linear embedding for each input asset
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_inputs)
        ])

        self.lstm = nn.LSTM(
            input_size=n_inputs * embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_inputs).

        Returns:
            Predictions of shape (batch, 1).
        """
        batch_size, seq_len, _ = x.shape

        # Apply per-asset embeddings
        embedded = []
        for i, emb in enumerate(self.embeddings):
            asset_input = x[:, :, i : i + 1]  # (batch, seq_len, 1)
            if self.ablate_idx is not None and i == self.ablate_idx:
                asset_input = torch.zeros_like(asset_input)
            embedded.append(emb(asset_input))  # (batch, seq_len, embed_dim)

        # Concatenate all embeddings along last dimension
        combined = torch.cat(embedded, dim=-1)  # (batch, seq_len, n_inputs*embed_dim)

        lstm_out, _ = self.lstm(combined)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        return self.output_head(last_hidden)  # (batch, 1)


def _create_sequences(
    data: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM training.

    Args:
        data: Array of shape (T, n_assets).
        history_len: Sequence length (look-back window).

    Returns:
        Tuple of (X, y) where X.shape=(N, history_len, n_assets) and y.shape=(N,).
    """
    n, n_assets = data.shape
    n_samples = n - history_len

    X = np.zeros((n_samples, history_len, n_assets))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = data[i : i + history_len]
        y[i] = data[i + history_len, 0]  # target is first column (target asset)

    return X, y


def neural_granger_test(
    returns: np.ndarray,
    target_idx: int,
    n_epochs: int = 200,
    history_len: int = 20,
    hidden_dim: int = 64,
    embed_dim: int = 16,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Test neural Granger causality from all assets to a target asset.

    Trains a ComponentLSTM on the full asset panel. Then for each source asset,
    ablates its embedding and measures the relative MSE increase on a validation
    set. A large MSE increase indicates strong Granger causality.

    Args:
        returns: Array of shape (T, n_assets). Column target_idx is the target.
        target_idx: Index of the target asset in the returns matrix.
        n_epochs: Training epochs.
        history_len: LSTM look-back window.
        hidden_dim: LSTM hidden dimension.
        embed_dim: Per-asset embedding dimension.
        learning_rate: Adam optimizer learning rate.
        val_fraction: Fraction of data for validation.
        batch_size: Mini-batch size.
        device: Torch device ('cpu' or 'cuda'). Auto-detected if None.

    Returns:
        Dict mapping source asset index to Granger causality score
        (relative MSE increase when ablated).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    returns = np.asarray(returns, dtype=np.float32)
    n_time, n_assets = returns.shape

    # Put target asset in first column for _create_sequences convention
    cols = [target_idx] + [i for i in range(n_assets) if i != target_idx]
    reordered = returns[:, cols]

    X_all, y_all = _create_sequences(reordered, history_len)

    # Train / validation split (chronological)
    n_val = max(1, int(len(X_all) * val_fraction))
    n_train = len(X_all) - n_val

    X_train = torch.tensor(X_all[:n_train], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_all[:n_train], dtype=torch.float32).to(device)
    X_val = torch.tensor(X_all[n_train:], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_all[n_train:], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    # ── Train full model ─────────────────────────────────────────────────────
    model = ComponentLSTM(
        n_inputs=n_assets,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            logger.debug("Epoch %d/%d, loss=%.4f", epoch + 1, n_epochs, loss.item())

    # ── Baseline MSE on validation set ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        baseline_pred = model(X_val).squeeze()
        baseline_mse = criterion(baseline_pred, y_val).item()

    logger.info("Baseline validation MSE: %.6f", baseline_mse)

    # ── Ablation test: zero out each input ───────────────────────────────────
    gc_scores: Dict[int, float] = {}

    for ablate_col in range(n_assets):
        # Map back to original index
        original_idx = cols[ablate_col]

        model.ablate_idx = ablate_col
        with torch.no_grad():
            ablated_pred = model(X_val).squeeze()
            ablated_mse = criterion(ablated_pred, y_val).item()

        if baseline_mse > 1e-10:
            gc_score = (ablated_mse - baseline_mse) / baseline_mse
        else:
            gc_score = 0.0

        gc_scores[original_idx] = max(0.0, gc_score)
        logger.debug(
            "GC score for asset %d: %.4f (ablated MSE: %.6f)",
            original_idx,
            gc_score,
            ablated_mse,
        )

    model.ablate_idx = None  # Reset
    return gc_scores
