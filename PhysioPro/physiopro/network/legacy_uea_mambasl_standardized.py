"""Train-stat standardized LSST execution for the pinned MambaSL adapter."""

from __future__ import annotations

import numpy as np
import torch

from .legacy_uea_mambasl_grouped import GroupedLengthMambaSLAdapter


def retained_channel_stats(dataset) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Estimate channel statistics from finite retained training rows only."""
    values = dataset.feature.to_numpy(dtype=np.float64).reshape(
        dataset.datasize, dataset.max_seq_len, dataset.num_variables
    )
    retained = np.isfinite(values).all(axis=-1)
    selected = values[retained]
    if selected.shape[0] < 2:
        raise ValueError("At least two retained training rows are required.")
    mean = selected.mean(axis=0)
    std = selected.std(axis=0, ddof=1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training statistics must be finite.")
    constant = std <= np.finfo(np.float32).eps
    std[constant] = 1.0
    return (
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
        int(constant.sum()),
    )


class TrainStandardizedGroupedMambaSLAdapter(GroupedLengthMambaSLAdapter):
    """Apply frozen train-only channel statistics before grouped execution."""

    execution_mode = "grouped_valid_length_train_standardized"

    def __init__(self, *args, input_mean: torch.Tensor, input_std: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("input_mean", input_mean.detach().clone().float())
        self.register_buffer("input_std", input_std.detach().clone().float())

    def forward(
        self,
        values: torch.Tensor,
        legacy_timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (values - self.input_mean) / self.input_std
        return super().forward(normalized, legacy_timestamps, valid)
