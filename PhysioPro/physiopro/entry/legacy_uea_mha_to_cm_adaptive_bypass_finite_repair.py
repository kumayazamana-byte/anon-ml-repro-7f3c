"""Finite numerical-repair entry point for three isolated LSST retries."""

from physiopro.network.mha_to_cm_lsst_finite_repair import install_finite_repair

from . import legacy_uea_mha_to_cm_adaptive_bypass as base


base.SUITE = "uea_legacy_train_es_mha_to_cm_adaptive_bypass_lsst_repair"
base.NUMERICAL_REPAIR = True


if __name__ == "__main__":
    install_finite_repair()
    base.main()
