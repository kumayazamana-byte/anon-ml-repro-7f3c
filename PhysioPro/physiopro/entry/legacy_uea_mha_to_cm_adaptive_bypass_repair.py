"""LSST-only numerical-repair entry for the adaptive/bypass factorial suite."""

from physiopro.network.mha_to_cm_numerical_repair import install_numerical_repair

from . import legacy_uea_mha_to_cm_adaptive_bypass as base


base.SUITE = "uea_legacy_train_es_mha_to_cm_adaptive_bypass_lsst_repair"
base.NUMERICAL_REPAIR = True


if __name__ == "__main__":
    install_numerical_repair()
    base.main()
