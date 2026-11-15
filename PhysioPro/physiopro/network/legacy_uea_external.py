"""Additional legacy-clock UEA baseline adapters.

These models intentionally consume the three tensors exposed by the frozen
``MaskTS(fill_nan_type="merge")`` path: values, retained timestamps, and a
right-padding mask.  They are isolated from the original TST/S5/MHA suite.

The original GRU-D formulation uses a feature-level observation mask and
feature-wise elapsed times.  ``MaskTS.merge`` removes that information, so the
implementation below is a documented row-level timestamp adaptation and must
be reported as ``GRU-D\N{DAGGER}``, not as an exact GRU-D reproduction.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .tsrnn import NETWORKS


def valid_tokens(mask: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Convert the legacy square padding mask into ``[batch, time]`` validity."""
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.ndim == 3:
        return ~mask.all(dim=-1)
    if mask.ndim == 2:
        return ~mask
    raise ValueError("mask must have shape [B, T] or [B, T, T].")


def legacy_deltas(times: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return elapsed time since the previous valid retained observation."""
    deltas = torch.zeros_like(times)
    previous = torch.zeros(times.shape[0], dtype=times.dtype, device=times.device)
    seen = torch.zeros(times.shape[0], dtype=torch.bool, device=times.device)
    for index in range(times.shape[1]):
        current = times[:, index]
        gap = (current - previous).clamp_min(0.0)
        deltas[:, index] = torch.where(valid[:, index] & seen, gap, torch.zeros_like(gap))
        previous = torch.where(valid[:, index], current, previous)
        seen = seen | valid[:, index]
    return deltas


def last_valid(sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    lengths = valid.long().sum(dim=1).clamp_min(1) - 1
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch, lengths]


def masked_sequence(sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid.unsqueeze(-1), sequence, torch.zeros_like(sequence))


def run_gru_cells(
    features: torch.Tensor,
    valid: torch.Tensor,
    cell: nn.GRUCell,
    hidden_size: int,
) -> torch.Tensor:
    """Run a GRUCell while freezing the state on right-padding tokens."""
    hidden = torch.zeros(features.shape[0], hidden_size, dtype=features.dtype, device=features.device)
    outputs = []
    for index in range(features.shape[1]):
        proposed = cell(features[:, index], hidden)
        hidden = torch.where(valid[:, index].unsqueeze(-1), proposed, hidden)
        outputs.append(torch.where(valid[:, index].unsqueeze(-1), hidden, torch.zeros_like(hidden)))
    return torch.stack(outputs, dim=1)


def rk4_integrate(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    func: nn.Module,
    step_size: float,
) -> torch.Tensor:
    """Integrate an autonomous neural ODE with a bounded, vectorized RK4 grid."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    max_delta = float(delta.detach().max().cpu())
    if max_delta <= 0.0:
        return hidden
    steps = max(1, int(math.ceil(max_delta / step_size)))
    step = delta / float(steps)
    state = hidden
    for _ in range(steps):
        k1 = func(state)
        k2 = func(state + 0.5 * step * k1)
        k3 = func(state + 0.5 * step * k2)
        k4 = func(state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


class LearnableTimeFeatures(nn.Module):
    """Learnable continuous time basis used by CADN and mTAN adaptations."""

    def __init__(self, output_size: int):
        super().__init__()
        if output_size < 2 or output_size % 2:
            raise ValueError("time feature size must be an even integer >= 2")
        half = output_size // 2
        self.frequency = nn.Parameter(torch.randn(half) * 0.2)
        self.phase = nn.Parameter(torch.zeros(half))
        self.output_size = output_size

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        angle = times.unsqueeze(-1) * self.frequency + self.phase
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


class LegacyGRUD(nn.Module):
    """Row-level elapsed-time GRU-D adaptation for the frozen merge interface."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        self.feature_mean = nn.Parameter(torch.zeros(input_size))
        self.input_decay = nn.Linear(1, input_size)
        self.hidden_decay = nn.Linear(1, hidden_size)
        self.input = nn.Linear(input_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        delta = legacy_deltas(t, valid)
        hidden = torch.zeros(x.shape[0], self.__hidden_size, dtype=x.dtype, device=x.device)
        outputs = []
        for index in range(x.shape[1]):
            gap = delta[:, index:index + 1]
            input_rate = F.softplus(self.input_decay(gap)) + 1e-4
            input_gamma = torch.exp(-gap * input_rate)
            imputed = input_gamma * x[:, index] + (1.0 - input_gamma) * self.feature_mean
            hidden_rate = F.softplus(self.hidden_decay(gap)) + 1e-4
            decayed_hidden = hidden * torch.exp(-gap * hidden_rate)
            proposed = self.cell(self.input(imputed), decayed_hidden)
            hidden = torch.where(valid[:, index].unsqueeze(-1), proposed, hidden)
            outputs.append(torch.where(valid[:, index].unsqueeze(-1), self.dropout(hidden), torch.zeros_like(hidden)))
        out = torch.stack(outputs, dim=1)
        return out, last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("gru_d_legacy")
class LegacyGRUDRegistered(LegacyGRUD):
    """Registry entry for the row-level GRU-D adaptation."""


class LegacyGRUDeltaT(nn.Module):
    """GRU with the retained legacy elapsed time as an explicit input."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        self.input = nn.Linear(input_size, hidden_size)
        self.delta = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())
        self.cell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        delta = legacy_deltas(t, valid)
        features = torch.cat((self.input(x), self.delta(delta.unsqueeze(-1))), dim=-1)
        out = run_gru_cells(features, valid, self.cell, self.__hidden_size)
        out = self.dropout(out)
        return masked_sequence(out, valid), last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("gru_delta_t_legacy")
class LegacyGRUDeltaTRegistered(LegacyGRUDeltaT):
    """Registry entry for the GRU-Delta-t adaptation."""


class _ODEFunction(nn.Module):
    def __init__(self, hidden_size: int, ode_hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ode_hidden_size),
            nn.Tanh(),
            nn.Linear(ode_hidden_size, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class LegacyODERNN(nn.Module):
    """ODE-RNN adaptation using RK4 flow followed by a GRU update."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        ode_hidden_size: int = 64,
        ode_step_size: float = 0.1,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        self.input = nn.Linear(input_size, hidden_size)
        self.ode = _ODEFunction(hidden_size, ode_hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.ode_step_size = float(ode_step_size)
        self.dropout = nn.Dropout(dropout)
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        delta = legacy_deltas(t, valid)
        hidden = torch.zeros(x.shape[0], self.__hidden_size, dtype=x.dtype, device=x.device)
        outputs = []
        for index in range(x.shape[1]):
            flowed = rk4_integrate(hidden, delta[:, index:index + 1], self.ode, self.ode_step_size)
            proposed = self.cell(self.input(x[:, index]), flowed)
            hidden = torch.where(valid[:, index].unsqueeze(-1), proposed, hidden)
            outputs.append(torch.where(valid[:, index].unsqueeze(-1), self.dropout(hidden), torch.zeros_like(hidden)))
        out = torch.stack(outputs, dim=1)
        return out, last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("ode_rnn_legacy")
class LegacyODERNNRegistered(LegacyODERNN):
    """Registry entry for the ODE-RNN adaptation."""


class LegacyNeuralCDE(nn.Module):
    """Controlled-midpoint Neural CDE adaptation over ``[t, x]`` increments."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        cde_hidden_size: int = 64,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        control_size = input_size + 1
        self.initial = nn.Linear(input_size, hidden_size)
        self.func = nn.Sequential(
            nn.Linear(hidden_size, cde_hidden_size),
            nn.Tanh(),
            nn.Linear(cde_hidden_size, hidden_size * control_size),
        )
        self.input_size = input_size
        self.control_size = control_size
        self.hidden_size_value = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

    def vector_field(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.func(hidden).view(hidden.shape[0], self.__hidden_size, self.control_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        hidden = torch.zeros(x.shape[0], self.__hidden_size, dtype=x.dtype, device=x.device)
        previous_x = torch.zeros_like(x[:, 0])
        previous_t = torch.zeros(x.shape[0], dtype=t.dtype, device=t.device)
        seen = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        outputs = []
        for index in range(x.shape[1]):
            current_valid = valid[:, index]
            current_x = x[:, index]
            current_t = t[:, index]
            initial_hidden = torch.tanh(self.initial(current_x))
            delta_x = current_x - previous_x
            delta_t = (current_t - previous_t).clamp_min(0.0).unsqueeze(-1)
            control_delta = torch.cat((delta_t, delta_x), dim=-1)
            field = self.vector_field(hidden)
            midpoint = hidden + 0.5 * torch.einsum("bhc,bc->bh", field, control_delta)
            candidate = hidden + torch.einsum("bhc,bc->bh", self.vector_field(midpoint), control_delta)
            proposed = torch.where(seen.unsqueeze(-1), candidate, initial_hidden)
            hidden = torch.where(current_valid.unsqueeze(-1), proposed, hidden)
            previous_x = torch.where(current_valid.unsqueeze(-1), current_x, previous_x)
            previous_t = torch.where(current_valid, current_t, previous_t)
            seen = seen | current_valid
            outputs.append(torch.where(current_valid.unsqueeze(-1), self.dropout(hidden), torch.zeros_like(hidden)))
        out = torch.stack(outputs, dim=1)
        return out, last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("neural_cde_legacy")
class LegacyNeuralCDERegistered(LegacyNeuralCDE):
    """Registry entry for the controlled-midpoint CDE adaptation."""


class LegacyCADN(nn.Module):
    """Attention-augmented ODE-RNN adaptation of the CADN family."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = 64,
        time_dim: int = 32,
        n_heads: int = 4,
        ode_hidden_size: int = 64,
        ode_step_size: float = 0.1,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads.")
        self.input = nn.Linear(input_size, hidden_size)
        self.time = LearnableTimeFeatures(time_dim)
        self.ode = _ODEFunction(hidden_size, ode_hidden_size)
        self.query = nn.Linear(input_size + hidden_size + time_dim, hidden_size)
        self.key = nn.Linear(hidden_size + time_dim, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.ode_step_size = float(ode_step_size)
        self.dropout = nn.Dropout(dropout)
        self.__output_size = hidden_size
        self.__hidden_size = hidden_size

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        delta = legacy_deltas(t, valid)
        hidden = torch.zeros(x.shape[0], self.__hidden_size, dtype=x.dtype, device=x.device)
        states = []
        state_times = []
        state_valid = []
        outputs = []
        scale = math.sqrt(float(self.head_dim))
        for index in range(x.shape[1]):
            flowed = rk4_integrate(hidden, delta[:, index:index + 1], self.ode, self.ode_step_size)
            time_now = self.time(t[:, index])
            states.append(flowed)
            state_times.append(time_now)
            state_valid.append(valid[:, index])
            history_h = torch.stack(states, dim=1)
            history_t = torch.stack(state_times, dim=1)
            history_valid = torch.stack(state_valid, dim=1)
            query = self.query(torch.cat((x[:, index], flowed, time_now), dim=-1))
            keys = self.key(torch.cat((history_h, history_t), dim=-1))
            values = self.value(history_h)
            query = query.view(x.shape[0], self.n_heads, self.head_dim)
            keys = keys.view(x.shape[0], -1, self.n_heads, self.head_dim)
            scores = torch.einsum("bnh,btnh->btn", query, keys) / scale
            scores = scores.masked_fill(~history_valid.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(scores, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0)
            context = torch.einsum("btn,btnh->bnh", weights, values.view(x.shape[0], -1, self.n_heads, self.head_dim))
            context = context.reshape(x.shape[0], self.__hidden_size)
            proposed = self.cell(torch.cat((self.input(x[:, index]), context), dim=-1), flowed)
            hidden = torch.where(valid[:, index].unsqueeze(-1), proposed, hidden)
            outputs.append(torch.where(valid[:, index].unsqueeze(-1), self.dropout(hidden), torch.zeros_like(hidden)))
        out = torch.stack(outputs, dim=1)
        return out, last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("cadn_legacy")
class LegacyCADNRegistered(LegacyCADN):
    """Registry entry for the attention-augmented ODE-RNN adaptation."""


class LegacyMTAN(nn.Module):
    """mTAN-style time-only attention over legacy retained timestamps."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        d_model: int = 64,
        time_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required.")
        if time_dim % n_heads:
            raise ValueError("time_dim must be divisible by n_heads.")
        self.value = nn.Linear(input_size, d_model)
        self.time = LearnableTimeFeatures(time_dim)
        self.n_heads = n_heads
        self.head_dim = time_dim // n_heads
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.__output_size = d_model
        self.__hidden_size = d_model

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        valid = valid_tokens(mask, x)
        values = self.value(x)
        basis = self.time(t).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        scores = torch.einsum("btnh,bsmh->btsnm", basis, basis)
        scores = scores.mean(dim=(-1, -2)) / math.sqrt(float(self.head_dim))
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.einsum("bts,bsd->btd", weights, values)
        out = values + self.dropout(context)
        out = self.output(out)
        out = masked_sequence(out, valid)
        return out, last_valid(out, valid)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("mtan_legacy")
class LegacyMTANRegistered(LegacyMTAN):
    """Registry entry for the mTAN adaptation."""
