"""Isolated LSST numerical-repair entry point for MHA-to-CM concat pooling."""

from physiopro.network.mha_to_cm_numerical_repair import install_numerical_repair

from . import legacy_uea_mha_to_cm as base


base.SUITE = "uea_legacy_train_es_mha_to_cm_concat_lsst_numerical_repair"


if __name__ == "__main__":
    install_numerical_repair()
    base.main()
