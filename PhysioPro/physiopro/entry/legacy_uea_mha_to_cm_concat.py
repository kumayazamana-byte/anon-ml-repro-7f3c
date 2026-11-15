"""Suite-identity wrapper around the frozen MHA-to-CM task entry point."""

from . import legacy_uea_mha_to_cm as base


base.SUITE = "uea_legacy_train_es_mha_to_cm_concat_pooling"


if __name__ == "__main__":
    base.main()
