# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torchcde
from .base import MODELS
from .timeseries import TS



@MODELS.register_module("mask_ts")
class MaskTS(TS):
    def __init__(
            self,
            task: str,
            optimizer: str,
            lr: float,
            weight_decay: float,
            loss_fn: str,
            metrics: List[str],
            observe: str,
            lower_is_better: bool,
            max_epochs: int,
            batch_size: int,
            network: Optional[nn.Module] = None,
            output_dir: Optional[Path] = None,
            checkpoint_dir: Optional[Path] = None,
            num_workers: Optional[int] = 8,
            early_stop: Optional[int] = None,
            early_stop_on_train: bool = False,
            out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
            model_path: Optional[str] = None,
            out_size: int = 1,
            aggregate: bool = True,
            fill_nan_type: Optional[str] = 'merge',
            norm_time_flg: Optional[bool] = True,
            pooling_mode: str = "last",
    ):
        """
        The model for general time-series prediction.

        Args:
            task: the prediction task, classification or regression.
            optimizer: which optimizer to use.
            lr: learning rate.
            weight_decay: L2 normlize weight
            loss_fn: loss function.
            metrics: metrics to evaluate model.
            observe: metric for model selection (earlystop).
            lower_is_better: whether a lower observed metric means better result.
            max_epochs: maximum epoch to learn.
            batch_size: batch size.
            early_stop: earlystop rounds.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            model_path: the path to existing model parameters for continued training or finetuning
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """
        if pooling_mode not in {"last", "mean", "concat_last_mean"}:
            raise ValueError("pooling_mode must be one of: last, mean, concat_last_mean.")
        output_size_multiplier = 2 if pooling_mode == "concat_last_mean" else 1
        super().__init__(
            task=task,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            metrics=metrics,
            observe=observe,
            lower_is_better=lower_is_better,
            max_epochs=max_epochs,
            batch_size=batch_size,
            network=network,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            num_workers=num_workers,
            early_stop=early_stop,
            early_stop_on_train=early_stop_on_train,
            out_ranges=out_ranges,
            model_path=model_path,
            out_size=out_size,
            aggregate=aggregate,
            output_size_multiplier=output_size_multiplier,
        )
        self.fill_nan_type = fill_nan_type
        self.norm_time_flg = norm_time_flg
        self.pooling_mode = pooling_mode

    @staticmethod
    def merge(sequences, pad_value=0.0, pad=False):
        device = sequences[0].device
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        if not pad:
            padded_seqs = torch.zeros(len(sequences), max(lengths), dim).to(device)
        else:
            padded_seqs = torch.zeros(len(sequences), max(lengths), dim).to(device) + pad_value

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def forward(self, inputs):
        if self.fill_nan_type == 'merge':   # for contiformer
            mask = inputs.isnan()
            tmp_mask = mask[..., 0]
            for i in range(1, mask.shape[-1]):
                tmp_mask = torch.bitwise_or(tmp_mask, mask[..., i])
            mask = tmp_mask

            X = []
            times = []
            for i in range(inputs.shape[0]):
                idx = torch.where(mask[i] == 0)
                X.append(inputs[i, idx[0].long(), :])
                times.append(idx[0].long().to(inputs.device).reshape(-1, 1))

            inputs, length = self.merge(X)
            times, _ = self.merge(times)
            times = times.squeeze(-1)

            if self.norm_time_flg:
                times = times / inputs.shape[1]

            attn_mask = torch.zeros(inputs.shape[0], inputs.shape[1], inputs.shape[1]).to(inputs.device)
            for i in range(inputs.shape[0]):
                attn_mask[i, :length[i], :length[i]] = torch.ones(length[i], length[i])
            attn_mask = (1 - attn_mask).bool()

            seq_out, _ = self.network(inputs, times, mask=attn_mask)  # [B, T, H]

            valid_lengths = torch.tensor(length, device=seq_out.device, dtype=torch.long)
            batch_idx = torch.arange(inputs.shape[0], device=seq_out.device)
            last_out = seq_out[batch_idx, valid_lengths - 1, :]
            if self.pooling_mode == "last":
                emb_outs = last_out
            else:
                token_idx = torch.arange(seq_out.shape[1], device=seq_out.device).unsqueeze(0)
                valid_token_mask = token_idx < valid_lengths.unsqueeze(1)
                mean_out = (
                    seq_out * valid_token_mask.unsqueeze(-1).to(seq_out.dtype)
                ).sum(dim=1) / valid_lengths.clamp_min(1).unsqueeze(-1).to(seq_out.dtype)
                if self.pooling_mode == "mean":
                    emb_outs = mean_out
                else:
                    emb_outs = torch.cat([last_out, mean_out], dim=-1)

        elif self.fill_nan_type == 'zero':  # for transformers and ssms
            inputs = torch.nan_to_num(inputs, nan=0.0)
            seq_out, emb_outs = self.network(inputs)  # [B, T, H]
        elif self.fill_nan_type in ['cubic', 'linear']:  # for cde-based, gru-based and ode-rnn
            if not self.norm_time_flg:
                times = torch.arange(inputs.shape[1]).to(inputs.device).float()
            else:
                times = torch.linspace(0, 1, inputs.shape[1]).to(inputs.device).float()
            if self.fill_nan_type == 'cubic':
                coeffs = torchcde.natural_cubic_spline_coeffs(inputs, t=times)
                spline = torchcde.CubicSpline(coeffs, t=times)
            else:
                coeffs = torchcde.linear_interpolation_coeffs(inputs, t=times)
                spline = torchcde.LinearInterpolation(coeffs, t=times)
            inputs = spline.evaluate(times)
            seq_out, emb_outs = self.network(inputs)  # [B, T, H]
        else:
            raise NotImplementedError

        if self.aggregate:
            out = emb_outs
        else:
            out = seq_out
        preds = self.act_out(self.fc_out(out).squeeze(-1))

        return preds
