from chainer import optimizer_hooks
from chainer import optimizers
from chainer import training
import numpy
import logging


logger = logging.getLogger(__name__)


def create_optimizer(configs):
    """
    :param optimizer_config: dict, 学習のパラメータを含む辞書
    """

    if "optimizer" not in configs:
        raise Exception("Optimizer configurations are not found")

    optimizer_configs = configs["optimizer"]
    optimizer_ = optimizer_configs["name"]
    optimizer_ = optimizer_.lower()

    if optimizer_ == "sgd":
        optimizer = optimizers.SGD(lr=optimizer_configs["learning_rate"])

    elif optimizer_ == "momentumsgd":
        optimizer = optimizers.MomentumSGD(lr=optimizer_configs["learning_rate"])

    elif optimizer_ == "adadelta":
        optimizer = optimizers.AdaDelta()

    elif optimizer_ == "adam":
        optimizer = optimizers.Adam(
            alpha=optimizer_configs["alpha"],
            beta1=optimizer_configs["beta1"],
            beta2=optimizer_configs["beta2"],
        )

    elif optimizer_ == "adabound":
        optimizer = optimizers.Adam(
            alpha=optimizer_configs["alpha"],
            beta1=optimizer_configs["beta1"],
            beta2=optimizer_configs["beta2"],
            adabound=True,
            final_lr=optimizer_configs["final_lr"],
        )  # NOQA

    else:
        raise Exception

    return optimizer


def add_hooks(optimizer, configs):
    """
    :param optimizer: chainer.Optimizer, chainerのオプティマイザ
    :param configs: pyner.util.config.ConfigParser
    """
    if "optimizer" not in configs:
        raise Exception("Optimizer configurations are not found")

    optimizer_configs = configs["optimizer"]

    if optimizer_configs.get("weight_decay"):
        logger.debug("\x1b[31mSet weight decay\x1b[0m")
        optimizer.add_hook(
            optimizer_hooks.WeightDecay(optimizer_configs["weight_decay"])
        )

    if "gradient_clipping" in optimizer_configs:
        clipping_threshold = optimizer_configs["gradient_clipping"]
        msg = "Enable gradient clipping:"
        msg += f" threshold \x1b[31m{clipping_threshold}\x1b[0m"
        logger.debug(msg)
        optimizer.add_hook(optimizer_hooks.GradientClipping(clipping_threshold))

    return optimizer


class LearningRateDecay(training.extension.Extension):

    """Exception to decay learning rate as in Ma+
    (http://www.aclweb.org/anthology/P16-1101)

    Learning rate would be updated to
    ``rate * / (1 + (1 + iteration)) * decay``

    This extension is also called before the training loop starts by default.
    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Exponent of polynomial shift.
        max_count (int): Number of this extension to be invoked.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.
    """

    invoke_before_training = True

    def __init__(self, attr, rate, decay, target=None, optimizer=None):
        self._attr = attr
        self._rate = rate
        self._decay = decay
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._rate)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self._rate / (1 + (self._decay * self._t))

        if self._target is not None:
            if self._rate > 0:
                # almost same as value = min(value, self._target), but this
                # line supports negative values, too
                if self._target / value > 1:
                    value = self._target
            else:
                # ditto
                if self._target / value < 1:
                    value = self._target

        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer("_t", self._t)
        self._last_value = serializer("_last_value", self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer("main")

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
