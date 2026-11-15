"""Isolated MHA-to-CM output-path interventions for the UEA factorial suite."""

from __future__ import annotations

from torch import nn

from .hippogp_contiformer import MhaToCmHiPPOGPAttention


class CmEnforcedMhaToCmAttention(nn.Module):
    """Keep MHA as a pre-write correction while removing its direct output bypass."""

    def __init__(self, source: MhaToCmHiPPOGPAttention) -> None:
        super().__init__()
        if not isinstance(source, MhaToCmHiPPOGPAttention):
            raise TypeError("CM-enforced conversion requires MhaToCmHiPPOGPAttention.")
        self.cm_attn = source.cm_attn
        self.local_attn = source.local_attn
        self.local_scale = source.local_scale

    def forward(self, x, t, mask=None, query_t=None):
        if query_t is not None and query_t.shape[1] != x.shape[1]:
            raise ValueError("mha_to_cm only supports token-aligned coordinate queries.")
        valid = self.cm_attn._valid_from_mask(mask, x)
        local, _ = self.local_attn(
            x, x, x, key_padding_mask=~valid, need_weights=False
        )
        local = self.local_scale * local
        return self.cm_attn(x + local, t, mask=mask, query_t=query_t)


def enforce_cm_output(network: nn.Module) -> nn.Module:
    """Replace only each ordered attention output rule; retain all parameters."""

    layers = getattr(getattr(network, "encoder", None), "layers", None)
    if layers is None:
        raise TypeError("Expected a HiPPO-GP ContiFormer encoder.")
    for layer in layers:
        layer.slf_attn = CmEnforcedMhaToCmAttention(layer.slf_attn)
    return network
