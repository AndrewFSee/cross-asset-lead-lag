"""Discovery module for information-theoretic and neural causality measures."""

from discovery.neural_granger import ComponentLSTM, neural_granger_test
from discovery.significance import bootstrap_te_significance, surrogate_significance
from discovery.time_lagged_mi import compute_tlmi_matrix, time_lagged_mi
from discovery.transfer_entropy import compute_te_matrix, transfer_entropy_knn

__all__ = [
    "transfer_entropy_knn",
    "compute_te_matrix",
    "time_lagged_mi",
    "compute_tlmi_matrix",
    "ComponentLSTM",
    "neural_granger_test",
    "bootstrap_te_significance",
    "surrogate_significance",
]
