# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from ..dataset import DATASETS, TSDataset, DfDataset
from ..model import MODELS
from ..network import NETWORKS


@configclass
class TrainConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    model: RegistryConfig[MODELS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset="train")
    valid_split_ratio = float(getattr(config.data, "dataset_split_ratio", 0.0) or 0.0)
    validset = None
    if valid_split_ratio > 0:
        validset = config.data.build(dataset="valid", preprocessor=trainset.preprocessor)
    testset = config.data.build(dataset="test", preprocessor=trainset.preprocessor)

    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )

    if config.data.type() in [TSDataset, DfDataset] and config.data.task == 'regression':
        out_size = 1
    else:
        out_size = trainset.num_classes

    model = config.model.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=out_size,
    )
    early_stop_on_train = bool(getattr(config.model, "early_stop_on_train", False))
    selection_set = validset
    if selection_set is None and not early_stop_on_train:
        selection_set = testset
    model.fit(trainset, selection_set, testset)
    model.predict(trainset, "train")
    if validset is not None:
        model.predict(validset, "valid")
    model.predict(testset, "test")


if __name__ == "__main__":
    _config = TrainConfig.fromcli()
    run_train(_config)
