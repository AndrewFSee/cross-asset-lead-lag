"""Lead-lag relationship monitoring and structural break detection."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StructuralBreakEvent:
    """A detected structural break in a lead-lag relationship.

    Attributes:
        timestamp: When the event was detected.
        event_type: Type of event ('te_spike', 'te_decay', 'regime_change').
        leader: Name of the leading asset.
        follower: Name of the following asset.
        old_value: Previous TE/metric value.
        new_value: New TE/metric value.
        description: Human-readable description of the event.
    """

    timestamp: datetime
    event_type: str
    leader: str
    follower: str
    old_value: float
    new_value: float
    description: str = field(default="")

    def __post_init__(self) -> None:
        if not self.description:
            self.description = (
                f"{self.event_type.upper()}: {self.leader} → {self.follower} "
                f"changed from {self.old_value:.4f} to {self.new_value:.4f}"
            )


class LeadLagMonitor:
    """Monitor rolling transfer entropy for structural breaks.

    Tracks the rolling history of TE values for each (leader, follower) pair.
    Detects z-score spikes (TE jumps above threshold) and decays (TE drops
    significantly). Also detects regime transitions from MS-VAR output.

    Args:
        te_spike_threshold: Z-score threshold for spike detection.
        te_decay_threshold: Fraction of rolling mean below which decay is flagged.
        rolling_window: Window size (in observations) for rolling statistics.
    """

    def __init__(
        self,
        te_spike_threshold: float = 2.0,
        te_decay_threshold: float = 0.5,
        rolling_window: int = 60,
    ) -> None:
        self.te_spike_threshold = te_spike_threshold
        self.te_decay_threshold = te_decay_threshold
        self.rolling_window = rolling_window

        # Rolling TE history: {(leader, follower): deque of values}
        self._te_history: Dict[Tuple[str, str], Deque[float]] = {}
        self._regime_history: Deque[int] = deque(maxlen=rolling_window)
        self._events: List[StructuralBreakEvent] = []
        self._last_te: Dict[Tuple[str, str], float] = {}
        self._last_regime: Optional[int] = None

    def update(
        self,
        te_matrix: pd.DataFrame,
        regime_probs: Optional[np.ndarray] = None,
        current_date: Optional[datetime] = None,
    ) -> List[StructuralBreakEvent]:
        """Update monitor with new TE matrix and regime probabilities.

        Args:
            te_matrix: Current TE matrix (rows=source, cols=target).
            regime_probs: Current regime probability vector (n_regimes,).
            current_date: Timestamp for detected events.

        Returns:
            List of newly detected StructuralBreakEvent instances.
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)

        new_events: List[StructuralBreakEvent] = []
        assets = te_matrix.index.tolist()

        # ── Check for TE spikes and decays ───────────────────────────────────
        for leader in assets:
            for follower in assets:
                if leader == follower:
                    continue

                pair = (leader, follower)
                current_te = float(te_matrix.loc[leader, follower])

                if pair not in self._te_history:
                    self._te_history[pair] = deque(maxlen=self.rolling_window)

                history = self._te_history[pair]
                history.append(current_te)

                if len(history) < 5:
                    self._last_te[pair] = current_te
                    continue

                arr = np.array(history)
                rolling_mean = arr[:-1].mean()
                rolling_std = arr[:-1].std()

                # Spike detection: current TE is z_score standard deviations above mean
                if rolling_std > 1e-10:
                    z_score = (current_te - rolling_mean) / rolling_std
                    if z_score > self.te_spike_threshold:
                        event = StructuralBreakEvent(
                            timestamp=current_date,
                            event_type="te_spike",
                            leader=leader,
                            follower=follower,
                            old_value=rolling_mean,
                            new_value=current_te,
                            description=(
                                f"TE spike detected: {leader} → {follower} "
                                f"(z={z_score:.2f}, TE={current_te:.4f})"
                            ),
                        )
                        new_events.append(event)
                        logger.info("TE spike: %s", event.description)

                # Decay detection: current TE dropped far below historical mean
                if rolling_mean > 1e-10 and current_te < rolling_mean * self.te_decay_threshold:
                    event = StructuralBreakEvent(
                        timestamp=current_date,
                        event_type="te_decay",
                        leader=leader,
                        follower=follower,
                        old_value=rolling_mean,
                        new_value=current_te,
                        description=(
                            f"TE decay detected: {leader} → {follower} "
                            f"(dropped to {current_te:.4f} from mean {rolling_mean:.4f})"
                        ),
                    )
                    new_events.append(event)
                    logger.info("TE decay: %s", event.description)

                self._last_te[pair] = current_te

        # ── Check for regime transitions ─────────────────────────────────────
        if regime_probs is not None:
            current_regime = int(np.argmax(regime_probs))
            self._regime_history.append(current_regime)

            if self._last_regime is not None and current_regime != self._last_regime:
                event = StructuralBreakEvent(
                    timestamp=current_date,
                    event_type="regime_change",
                    leader="MARKET",
                    follower="MARKET",
                    old_value=float(self._last_regime),
                    new_value=float(current_regime),
                    description=(f"Regime transition: {self._last_regime} → {current_regime}"),
                )
                new_events.append(event)
                logger.info("Regime change: %s", event.description)

            self._last_regime = current_regime

        self._events.extend(new_events)
        return new_events

    @property
    def events(self) -> List[StructuralBreakEvent]:
        """Full event history."""
        return self._events.copy()
