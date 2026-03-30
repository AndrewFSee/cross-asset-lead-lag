"""Tests for the lead-lag monitor and structural break detection."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from agent.monitor import LeadLagMonitor, StructuralBreakEvent


class TestStructuralBreakEvent:
    """Tests for StructuralBreakEvent."""

    def test_auto_description(self):
        """Auto-generated description should include event type and assets."""
        event = StructuralBreakEvent(
            timestamp=datetime(2024, 1, 1),
            event_type="te_spike",
            leader="VIX",
            follower="HY_OAS",
            old_value=0.02,
            new_value=0.08,
        )
        assert "TE_SPIKE" in event.description
        assert "VIX" in event.description
        assert "HY_OAS" in event.description

    def test_custom_description(self):
        """Custom description should override auto-generated one."""
        event = StructuralBreakEvent(
            timestamp=datetime(2024, 1, 1),
            event_type="te_spike",
            leader="A",
            follower="B",
            old_value=0.0,
            new_value=1.0,
            description="Custom message",
        )
        assert event.description == "Custom message"


class TestLeadLagMonitor:
    """Tests for LeadLagMonitor."""

    def _make_te_matrix(self, assets, values=None):
        """Helper to create a TE matrix DataFrame."""
        n = len(assets)
        if values is None:
            values = np.zeros((n, n))
        return pd.DataFrame(values, index=assets, columns=assets)

    def test_no_events_initially(self):
        """First few updates should not produce events (building history)."""
        monitor = LeadLagMonitor(rolling_window=10)
        assets = ["A", "B"]
        te = self._make_te_matrix(assets, [[0.0, 0.05], [0.03, 0.0]])
        events = monitor.update(te)
        assert len(events) == 0

    def test_spike_detection(self):
        """A sudden TE increase should trigger a te_spike event."""
        monitor = LeadLagMonitor(te_spike_threshold=2.0, rolling_window=20)
        assets = ["A", "B"]
        rng = np.random.default_rng(42)

        # Build up history with small natural variance around 0.05
        for _ in range(15):
            base_val = 0.05 + rng.normal(0, 0.005)
            te = self._make_te_matrix(assets, [[0.0, base_val], [0.03, 0.0]])
            monitor.update(te)

        # Inject a large spike (10x baseline)
        spike_te = self._make_te_matrix(assets, [[0.0, 1.00], [0.03, 0.0]])
        events = monitor.update(spike_te)

        spike_events = [e for e in events if e.event_type == "te_spike"]
        assert len(spike_events) > 0
        assert spike_events[0].leader == "A"
        assert spike_events[0].follower == "B"

    def test_decay_detection(self):
        """A TE drop below threshold fraction should trigger a te_decay event."""
        monitor = LeadLagMonitor(te_decay_threshold=0.5, rolling_window=10)
        assets = ["A", "B"]

        # Build up stable history with higher TE
        for _ in range(8):
            te = self._make_te_matrix(assets, [[0.0, 0.10], [0.08, 0.0]])
            monitor.update(te)

        # Inject a decay
        decay_te = self._make_te_matrix(assets, [[0.0, 0.02], [0.01, 0.0]])
        events = monitor.update(decay_te)

        decay_events = [e for e in events if e.event_type == "te_decay"]
        assert len(decay_events) > 0

    def test_regime_change_detection(self):
        """Regime transition should trigger a regime_change event."""
        monitor = LeadLagMonitor()
        assets = ["A", "B"]
        te = self._make_te_matrix(assets, [[0.0, 0.05], [0.03, 0.0]])

        # First update: set regime to 0
        monitor.update(te, regime_probs=np.array([0.8, 0.2]))

        # Second update: regime switches to 1
        events = monitor.update(te, regime_probs=np.array([0.2, 0.8]))

        regime_events = [e for e in events if e.event_type == "regime_change"]
        assert len(regime_events) == 1
        assert regime_events[0].old_value == 0.0
        assert regime_events[0].new_value == 1.0

    def test_events_property_returns_copy(self):
        """events property should return a copy (not a reference)."""
        monitor = LeadLagMonitor()
        assets = ["A", "B"]
        te = self._make_te_matrix(assets)
        monitor.update(te)
        e1 = monitor.events
        e2 = monitor.events
        assert e1 is not e2
