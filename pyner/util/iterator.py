import chainer.iterators as It
from pyner.named_entity.dataset import SequenceLabelingDataset


def create_iterator(vocab, configs, mode, transform):
    if "iteration" not in configs:
        raise Exception("Batch configurations are not found")

    if "external" not in configs:
        raise Exception("External data configurations are not found")

    is_train = (mode == "train")
    iteration_configs = configs["iteration"]
    external_configs = configs["external"]

    dataset = SequenceLabelingDataset(vocab, external_configs, mode, transform)
    batch_size = iteration_configs["batch_size"] if is_train else len(dataset)

    iterator = It.SerialIterator(
        dataset,
        batch_size=batch_size,
        repeat=is_train,
        shuffle=is_train
    )
    return iterator
