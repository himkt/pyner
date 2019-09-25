from pyner.named_entity.dataset import update_instances


def dataset():
    return [
        [
            ["I", "O"],
            ["have", "O"],
            ["an", "O"],
            ["apple", "O"],
            ["made", "O"],
            ["in", "O"],
            ["Japan", "S-LOC"],
        ],
        [
            ["You", "O"],
            ["live", "O"],
            ["in", "O"],
            ["Japan", "S-LOC"],
        ],
        [
            ["You", "O"],
            ["live", "O"],
            ["in", "O"],
            ["United", "B-LOC"],
            ["States", "E-LOC"],
        ],
        [
            ["himkt", "S-PER"],
            ["lives", "O"],
            ["in", "O"],
            ["Tokyo", "S-LOC"],
        ],
        [
            ["himkt", "S-PER"],
            ["likes", "O"],
            ["taking", "O"],
            ["photos", "O"],
            ["in", "O"],
            ["France", "S-LOC"],
        ],
    ]


def expect():
    return [
        [
            ('I', 'have', 'an', 'apple', 'made', 'in', 'Japan'),
            ('You', 'live', 'in', 'Japan'),
            ('You', 'live', 'in', 'United', 'States'),
            ('himkt', 'lives', 'in', 'Tokyo'),
            ('himkt', 'likes', 'taking', 'photos', 'in', 'France')
        ],
        [
            ('O', 'O', 'O', 'O', 'O', 'O', 'S-LOC'),
            ('O', 'O', 'O', 'S-LOC'),
            ('O', 'O', 'O', 'B-LOC', 'E-LOC'),
            ('S-PER', 'O', 'O', 'S-LOC'),
            ('S-PER', 'O', 'O', 'O', 'O', 'S-LOC')
        ]
    ]


def test_update_instances_on_train():
    expect_percentile = [e[:4] for e in expect()]
    assert expect() == update_instances(dataset(), {"train_size": 1.0}, "train")  # NOQA
    assert expect_percentile == update_instances(dataset(), {"train_size": 0.8}, "train")  # NOQA


def test_update_instances_on_valid():
    assert expect() == update_instances(dataset(), {"valid_size": 1.0}, "valid")  # NOQA
    assert expect() == update_instances(dataset(), {"valid_size": 0.8}, "valid")  # NOQA


def test_update_instances_on_test():
    assert expect() == update_instances(dataset(), {"test_size": 1.0}, "test")
    assert expect() == update_instances(dataset(), {"test_size": 0.8}, "test")
