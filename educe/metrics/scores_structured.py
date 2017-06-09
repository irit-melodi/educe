"""Classification metrics for structured outputs.

"""

from collections import Counter
from itertools import chain, izip

import numpy as np


def _unique_labels(y):
    """Set of unique labels in y"""
    return set(y_ij[1] for y_ij in
               chain.from_iterable(y_i for y_i in y))


def unique_labels(*ys):
    """Extract an ordered array of unique labels.

    Parameters
    ----------
    elt_type: string
        Type of each element, determines how to find the label

    See also
    --------
    This is the structured version of
    `sklearn.utils.multiclass.unique_labels`
    """
    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))
    # TODO check the set of labels contains a unique (e.g. string) type
    # of values
    return np.array(sorted(ys_labels))


def precision_recall_fscore_support(y_true, y_pred, labels=None,
                                    average=None, return_support_pred=True):
    """Compute precision, recall, F-measure and support for each class.

    The support is the number of occurrences of each class in
    ``y_true``.

    This is essentially a structured version of
    sklearn.metrics.classification.precision_recall_fscore_support .

    It should apply equally well to lists of constituency tree spans
    and lists of dependency edges.

    Parameters
    ----------
    y_true: list of iterable
        Ground truth target structures, encoded in a sparse format (e.g.
        list of edges or span descriptions).

    y_pred: list of iterable
        Estimated target structures, encoded in a sparse format (e.g. list
        of edges or span descriptions).

    labels: list, optional
        The set of labels to include, and their order if ``average is
        None``.

    average: string, [None (default), 'binary', 'micro', 'macro']
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the positive class.
            This is applicable only if targets are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true
            positives, false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account.

    return_support_pred: boolean, True by default
        If True, output the support of the prediction. This is useful
        for structured prediction because y_true and y_pred can differ
        in length.

    Returns
    -------
    precision: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    recall: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    fscore: float (if average is not None) or array of float, shape=\
        [n_unique_labels]

    support: int (if average is not None) or array of int, shape=\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.

    support_pred: int (if average is not None) or array of int, shape=\
        [n_unique_labels], if ``return_support_pred``.
        If The number of occurrences of each label in ``ctree_pred``.
    """
    average_options = frozenset([None, 'micro', 'macro'])
    if average not in average_options:
        raise ValueError('average has to be one of' +
                         str(average_options))
    # TMP
    if average == 'macro':
        raise NotImplementedError('average currently has to be micro or None')
    # end TMP

    # gather an ordered list of unique labels from y_true and y_pred
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        # n_labels = None
    else:
        # EXPERIMENTAL
        labels = [lbl for lbl in labels if lbl in present_labels]
        # n_labels = len(labels)
        # FIXME complete/fix this
        # raise ValueError('Parameter `labels` is currently unsupported')
        # end EXPERIMENTAL

    # compute tp_sum, pred_sum, true_sum
    # true positives for each tree
    tp = [set(yi_true) & set(yi_pred)
          for yi_true, yi_pred in izip(y_true, y_pred)]

    # TODO find a nicer and faster design that resembles sklearn's, e.g.
    # use np.bincount instead of collections.Counter
    tp_sum = Counter(y_ij[1] for y_ij in chain.from_iterable(tp))
    true_sum = Counter(y_ij[1] for y_ij in chain.from_iterable(y_true))
    pred_sum = Counter(y_ij[1] for y_ij in chain.from_iterable(y_pred))
    # transform to np arrays of floats
    tp_sum = np.array([float(tp_sum[lbl]) for lbl in labels])
    true_sum = np.array([float(true_sum[lbl]) for lbl in labels])
    pred_sum = np.array([float(pred_sum[lbl]) for lbl in labels])

    # TODO rewrite to compute by summing over scores broken down by label
    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])

    # finally compute the desired statistics
    # when the div denominator is 0, assign 0.0 (instead of np.inf)
    precision = tp_sum / pred_sum
    precision[pred_sum == 0] = 0.0

    recall = tp_sum / true_sum
    recall[true_sum == 0] = 0.0

    f_score = 2 * (precision * recall) / (precision + recall)
    f_score[precision + recall == 0] = 0.0

    if average is not None:
        precision = np.average(precision)
        recall = np.average(recall)
        f_score = np.average(f_score)
        true_sum = np.average(true_sum)  # != sklearn: we keep the support
        pred_sum = np.average(pred_sum)

    if return_support_pred:
        return precision, recall, f_score, true_sum, pred_sum
    else:
        return precision, recall, f_score, true_sum
