import chainer
import collections
import copy
import numpy


def evaluate(n_test, n_pred, n_relv):
    if n_pred * n_test == 0:
        precision = float('nan')
        recall = float('nan')
        f1_score = float('nan')

        if n_pred != 0:
            recall = n_relv / n_pred

        if n_test != 0:
            precision = n_relv / n_test

    else:
        precision = n_relv / n_test
        recall = n_relv / n_pred

        if precision * recall == 0:
            f1_score = float('nan')

        else:
            numerator = 2 * precision * recall
            denominator = precision + recall
            f1_score = numerator / denominator

    return precision, recall, f1_score


def evaluate_accuracy(y_test, y_pred):
    sum_cnt = 0
    correct_cnt = 0

    for (gold, pred) in zip(y_test, y_pred):
        sum_cnt += len(gold)
        correct_cnt += sum(gold == pred)

    sum_cnt = 1 if sum_cnt == 0 else sum_cnt
    accuracy = float(correct_cnt) / sum_cnt
    accuracy = accuracy
    return accuracy


class LabeledRange:

    def __init__(self, sent_id, label, start, last):
        self.sent_id = sent_id
        self.label = label
        self.start = start
        self.last = last

    def __eq__(self, another):
        if self.sent_id != another.sent_id:
            return False

        if self.label != another.label:
            return False

        if self.start != another.start:
            return False

        if self.last != another.last:
            return False

        return True

    def __repr__(self):
        return f'({self.sent_id}-{self.label}, ({self.start}-{self.last}))'

    def __str__(self):
        return f'{self.sent_id}, {self.label}, {self.start}-{self.last}'


class NEREvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, target, converter, device, label2index):
        super(NEREvaluator, self).__init__(iterator, target, converter, device)  # NOQA
        self.index2label = {index: label for label, index in label2index.items()}  # NOQA
        self.labels = sorted(list(label2index.keys()))

    def to_labeledRange(self, result):
        BEGIN = 0
        INSIDE = 1
        OUTSIDE = 2
        state = OUTSIDE

        lr, label = None, None
        lr_result = [[] for _ in range(len(result))]

        for i in range(len(result)):
            _result = result[i]
            for k in range(len(_result)):
                state_ = _result[k][0]

                if state_ == 'B':
                    state = BEGIN

                elif state_ == 'I':
                    state = INSIDE

                elif state_ == 'O':
                    state = OUTSIDE

                label = _result[k][2:]

                if state == BEGIN:
                    if lr is not None:
                        lr_result[i].append(lr)

                    lr = LabeledRange(sent_id=i, label=label, start=k, last=k)

                elif state == INSIDE:
                    if lr is not None and lr.label == label:
                        lr.last += 1

                    else:
                        if lr is not None:
                            lr_result[i].append(lr)

                        lr = LabeledRange(sent_id=i, label=label, start=k, last=k)  # NOQA

                elif state == OUTSIDE:
                    if lr is not None:
                        lr_result[i].append(lr)
                        lr = None

                else:
                    assert(False)

            if lr is not None:
                lr_result[i].append(lr)
                lr = None

        return lr_result

    def evaluate(self):
        iterator = self.get_iterator('main')
        target = self.get_target('main')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()
        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                with chainer.function.no_backprop_mode():
                    *batch, labelids = self.converter(batch, self.device)
                    y_pred = target.predict(*batch)
                    y_pred = numpy.asarray([numpy.asarray(list(map(lambda x: self.index2label[x], chainer.cuda.to_cpu(y)))) for y in y_pred])  # NOQA
                    y_pred_lr_list = sum(self.to_labeledRange(y_pred), [])

                    y_test = labelids
                    y_test = numpy.asarray([numpy.asarray(list(map(lambda x: self.index2label[x], chainer.cuda.to_cpu(y)))) for y in y_test])  # NOQA
                    y_test_lr_list = sum(self.to_labeledRange(y_test), [])

                    pred_counter = collections.Counter([x.label for x in y_pred_lr_list])  # NOQA
                    test_counter = collections.Counter([x.label for x in y_test_lr_list])  # NOQA
                    relv_counter = {}

                    is_in_test = {str(x): 1 for x in y_test_lr_list}

                    for y_pred_lr in y_pred_lr_list:
                        key = str(y_pred_lr)
                        label = y_pred_lr.label

                        if key in is_in_test.keys():
                            relv_counter[label] = relv_counter.get(label, 0) + 1  # NOQA

                    sum_n_test = sum(test_counter.values())
                    sum_n_pred = sum(pred_counter.values())
                    sum_n_relv = sum(relv_counter.values())
                    result = evaluate(sum_n_test, sum_n_pred, sum_n_relv)
                    precision, recall, f1_score = result
                    accuracy = evaluate_accuracy(y_test, y_pred)

                    chainer.reporter.report({'ner_precision': precision}, target)  # NOQA
                    chainer.reporter.report({'ner_recall': recall}, target)
                    chainer.reporter.report({'ner_f1_score': f1_score}, target)
                    chainer.reporter.report({'ner_accuracy': accuracy}, target)

                    labelset = [label[2:] for label in self.index2label.values()]  # NOQA
                    labelset = set(labelset) - {'O', ''}

                    for label in labelset:
                        n_test = test_counter.get(label, 0)
                        n_pred = pred_counter.get(label, 0)
                        n_relv = relv_counter.get(label, 0)
                        result = evaluate(n_test, n_pred, n_relv)
                        precision, recall, f1_score = result

                        chainer.reporter.report({f'{label}_P': precision}, target)  # NOQA
                        chainer.reporter.report({f'{label}_R': recall}, target)  # NOQA
                        chainer.reporter.report({f'{label}_F': f1_score}, target)  # NOQA
                        chainer.reporter.report({f'{label}_S': n_test}, target)  # NOQA

                summary.add(observation)
        return summary.compute_mean()
