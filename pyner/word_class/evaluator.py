from chainer import function
from chainer import reporter as reporter_module
from chainer.training.extensions import Evaluator
from sklearn.metrics import f1_score

import copy


class ClassificationEvaluator(Evaluator):

    def __init__(self, iterator, target,
                 transform_func, converter, device=-1):

        super(ClassificationEvaluator, self).\
            __init__(iterator, target, converter, device)

        self.transform_func = transform_func
        self.device = device

    def evaluate(self):
        iterator = self.get_iterator('main')
        target = self.get_target('main')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                with function.no_backprop_mode():
                    in_arrays, t_arrays = self.converter(batch, self.device)

                    p_arrays = target.predict(in_arrays)
                    _, t_labels = list(zip(*self.transform_func(
                        in_arrays, t_arrays)))
                    _, p_labels = list(zip(*self.transform_func(
                        in_arrays, p_arrays)))

                    fscore = f1_score(t_labels, p_labels, average='micro')
                    reporter_module.report(
                        {'loss': target(in_arrays, t_arrays)}, target)
                    reporter_module.report({'fscore': fscore}, target)

            summary.add(observation)

        return summary.compute_mean()
