"""Isolated memory repair entry point for the failed t-PatchGNN UEA cells."""

from physiopro.network.legacy_uea_tpatchgnn_memory_efficient import (
    TPatchGNNMemoryEfficientLegacyUEAAdapter,
)

from . import legacy_uea_tpatchgnn as base


base.SUITE = "uea_legacy_train_es_tpatchgnn_memory_repair"
base.TPatchGNNLegacyUEAAdapter = TPatchGNNMemoryEfficientLegacyUEAAdapter


if __name__ == "__main__":
    base.main()
