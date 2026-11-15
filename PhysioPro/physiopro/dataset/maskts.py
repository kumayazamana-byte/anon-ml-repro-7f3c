# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The dataloader for loading .ts data.
Refer to the code in https://github.com/gzerveas/mvts_transformer.
"""
from typing import Optional, Union
import hashlib
import numpy as np
from utilsd.config import ClassConfig
from .base import DATASETS
from .ts import TSDataset
from .preprocessor import Preprocessor


@DATASETS.register_module("mask_ts")
class MaskTSDataset(TSDataset):
    """
    A dataset for time series forecasting.
    """

    def __init__(
            self,
            prefix: str = "data/Multivariate_ts",
            name: str = "Heartbeat",
            max_seq_len: int = 0,
            dataset: str = "train",
            dataset_split_ratio: float = 0.,
            task: str = "classification",
            preprocessor: Optional[Union[ClassConfig[Preprocessor], Preprocessor]] = None,
            mask_ratio: Optional[float] = 0.3,
            mask_seed: Optional[int] = None,
    ):
        super().__init__(prefix, name, max_seq_len, dataset, dataset_split_ratio, task, preprocessor)

        self.mask_ratio = mask_ratio
        self.mask_seed = mask_seed
        rng = None
        if mask_seed is not None:
            seed_key = f"{name}:{dataset}:{mask_ratio}:{mask_seed}".encode("utf-8")
            split_offset = int(hashlib.md5(seed_key).hexdigest()[:8], 16)
            rng = np.random.default_rng((int(mask_seed) + split_offset) % (2 ** 32))

        for i in range(self.datasize):
            _length = self._max_seq_len - self.feature.loc[i]['dim_0'].isnull().sum()
            num_removed = int(_length * self.mask_ratio)
            if rng is None:
                removed_points = np.random.choice(_length, num_removed, replace=False)
            else:
                removed_points = rng.choice(_length, num_removed, replace=False)
            removed_points = removed_points + self.feature.loc[i]['dim_0'].isnull().sum()
            for idx in removed_points:
                self.feature.loc[i].iloc[idx] = np.nan
