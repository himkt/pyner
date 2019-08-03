from pyner.named_entity.dataset import SequenceLabelingDataset

import chainer.iterators as It


def create_iterator(vocab, configs, role, transform):
    if "iteration" not in configs:
        raise Exception("Batch configurations are not found")

    if "external" not in configs:
        raise Exception("External data configurations are not found")

    iteration_configs = configs["iteration"]
    external_configs = configs["external"]

    is_train = role == "train"
    shuffle = True if is_train else False
    repeat = True if is_train else False

    dataset = SequenceLabelingDataset(vocab, external_configs, role, transform)
    batch_size = iteration_configs["batch_size"] if is_train else len(dataset)

    iterator = It.SerialIterator(
        dataset, batch_size=batch_size, repeat=repeat, shuffle=shuffle
    )
    return iterator
