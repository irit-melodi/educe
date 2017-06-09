"""PARSEVAL metrics adapted for RST constituency trees.

References
----------
.. [1] `Daniel Marcu (2000). "The theory and practice of discourse
       parsing and summarization." MIT press.

"""

from __future__ import absolute_import, print_function

from educe.metrics.parseval import (parseval_scores, parseval_report,
                                    parseval_compact_report,
                                    parseval_detailed_report,
                                    parseval_similarity)


# label extraction functions
LBL_FNS = [
    ('S', lambda span: 1),
    ('N', lambda span: span[1]),
    ('R', lambda span: span[2]),
    ('F', lambda span: '{}-{}'.format(span[2], span[1])),
    # WIP 2016-11-10 add head to evals
    ('S+H', lambda span: span[3]),
    ('N+H', lambda span: '{}-{}'.format(span[1], span[3])),
    ('R+H', lambda span: '{}-{}'.format(span[2], span[3])),
    ('F+H', lambda span: '{}-{}'.format(span[2], span[1])),
    # end WIP head
]


def rst_parseval_scores(ctree_true, ctree_pred, lbl_fn, subtree_filter=None,
                        labels=None, average=None):
    """Compute RST PARSEVAL scores for ctree_pred wrt ctree_true.

    Notably, the root node of both ctrees is excluded from the scoring
    procedure.

    Parameters
    ----------
    ctree_true : list of list of RSTTree or SimpleRstTree
        List of reference RST trees, one per document.

    ctree_pred : list of list of RSTTree or SimpleRstTree
        List of predicted RST trees, one per document.

    lbl_fn : function, optional
        Function to relabel spans.

    subtree_filter : function, optional
        Function to filter all local trees.

    labels : list of string, optional
        Corresponds to sklearn's target_names IMO

    average : one of {'micro', 'macro'}, optional
        TODO, see scores_structured

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Weighted average of the precision of each class.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.

    """
    return parseval_scores(ctree_true, ctree_pred,
                           subtree_filter=subtree_filter,
                           exclude_root=True, lbl_fn=lbl_fn,
                           labels=labels, average=average)


def rst_parseval_compact_report(parser_true, parser_preds,
                                ctree_type='RST', subtree_filter=None,
                                span_type='edus',
                                metric_types=None, digits=4,
                                percent=False,
                                print_support=True,
                                per_doc=False,
                                add_trivial_spans=False,
                                stringent=False):
    """Build a compact text report showing the f1-scores of the PARSEVAL
    discourse metrics.

    This is the simplest report we need to generate, it corresponds
    to the arrays of results from the literature.
    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    parser_true: str
        Name of the parser used as a ref.

    parser_preds: List of (parser_name, List of ctree_pred)
        List of predictions for each parser.

    ctree_type : one of {'RST', 'SimpleRST'}, defaults to 'RST'
        Type of ctrees considered in the evaluation procedure.
        'RST' is the standard type of ctrees used in the RST corpus,
        it triggers the exclusion of the root node from the evaluation
        but leaves are kept.
        'SimpleRST' is a binarized variant of RST trees where each
        internal node corresponds to an attachment decision ; in other
        words, it is a binary ctree where the nuclearity and relation label
        are moved one node up compared to the standard RST trees. This
        triggers the exclusion of leaves from the eval, but the root node
        is kept.

    subtree_filter: function, optional
        Function to filter all local trees.

    metric_types : list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'N', 'R', 'F'].

    digits : int, defaults to 4
        Number of decimals to print.

    print_support : boolean, defaults to True
        If True, the true support, i.e. the number of reference spans,
        is also displayed. This is useful to control whether the
        reference ctrees have been binarized.

    per_doc : boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.

    add_trivial_spans : boolean, defaults to False
        If True, trivial spans 0-0, 0-n, 1-n are added ; this is meant to
        replicate the evaluation procedure of Li et al.'s dependency RST
        parser.

    stringent : boolean, defaults to False
        TODO
    """
    # filter root or leaves, depending on the type of ctree
    if ctree_type not in ['RST', 'SimpleRST']:
        raise ValueError("ctree_type should be one of {'RST', 'SimpleRST'}")
    if ctree_type == 'RST':
        # standard RST ctree: exclude root
        exclude_root = True
        subtree_filter = subtree_filter
    elif ctree_type == 'SimpleRST':
        # SimpleRST variant: keep root, exclude leaves
        exclude_root = False  # TODO try True first, should get same as before
        not_leaf = lambda t: t.height() > 2  # TODO unit test!
        if subtree_filter is None:
            subtree_filter = not_leaf
        else:
            subtree_filter = lambda t: not_leaf(t) and subtree_filter(t)

    # select metrics and the corresponding functions
    if metric_types is None:
        # metric_types = ['S', 'N', 'R', 'F']
        metric_types = [x[0] for x in LBL_FNS]
    if set(metric_types) - set(x[0] for x in LBL_FNS):
        raise ValueError('Unknown metric types in {}'.format(metric_types))
    metric2lbl_fn = dict(LBL_FNS)
    lbl_fns = [(metric_type, metric2lbl_fn[metric_type])
               for metric_type in metric_types]

    return parseval_compact_report(parser_true, parser_preds,
                                   exclude_root=exclude_root,
                                   subtree_filter=subtree_filter,
                                   lbl_fns=lbl_fns,
                                   span_type=span_type,
                                   digits=digits,
                                   percent=percent,
                                   print_support=print_support,
                                   per_doc=per_doc,
                                   add_trivial_spans=add_trivial_spans)


def rst_parseval_similarity(parser_preds,
                            ctree_type='RST',
                            subtree_filter=None,
                            span_type='edus',
                            metric_type='S', digits=4,
                            percent=False,
                            print_support=True,
                            per_doc=False,
                            add_trivial_spans=False,
                            stringent=False,
                            out_format='str'):
    """Build a similarity matrix showing the f1-scores of a PARSEVAL
    discourse metric.

    This is the simplest report we need to generate, it corresponds
    to the arrays of results from the literature.
    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    parser_preds : List of (parser_name, List of ctree_pred)
        List of predictions for each parser.

    ctree_type : one of {'RST', 'SimpleRST'}, defaults to 'RST'
        Type of ctrees considered in the evaluation procedure.
        'RST' is the standard type of ctrees used in the RST corpus,
        it triggers the exclusion of the root node from the evaluation
        but leaves are kept.
        'SimpleRST' is a binarized variant of RST trees where each
        internal node corresponds to an attachment decision ; in other
        words, it is a binary ctree where the nuclearity and relation label
        are moved one node up compared to the standard RST trees. This
        triggers the exclusion of leaves from the eval, but the root node
        is kept.

    subtree_filter : function, optional
        Function to filter all local trees.

    metric_type : str
        Metric on which similarity is judged, defaults to 'S'.

    digits : int, defaults to 4
        Number of decimals to print.

    print_support : boolean, defaults to True
        If True, the true support, i.e. the number of reference spans,
        is also displayed. This is useful to control whether the
        reference ctrees have been binarized.

    per_doc : boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.

    add_trivial_spans : boolean, defaults to False
        If True, trivial spans 0-0, 0-n, 1-n are added ; this is meant to
        replicate the evaluation procedure of Li et al.'s dependency RST
        parser.

    stringent : boolean, defaults to False
        TODO

    out_format : str, one of {'str', 'latex'}
        Output format.
    """
    # filter root or leaves, depending on the type of ctree
    if ctree_type not in ['RST', 'SimpleRST']:
        raise ValueError("ctree_type should be one of {'RST', 'SimpleRST'}")
    if ctree_type == 'RST':
        # standard RST ctree: exclude root
        exclude_root = True
        subtree_filter = subtree_filter
    elif ctree_type == 'SimpleRST':
        # SimpleRST variant: keep root, exclude leaves
        exclude_root = False  # TODO try True first, should get same as before
        not_leaf = lambda t: t.height() > 2  # TODO unit test!
        if subtree_filter is None:
            subtree_filter = not_leaf
        else:
            subtree_filter = lambda t: not_leaf(t) and subtree_filter(t)

    # select metrics and the corresponding functions
    metric_types = ['S', 'N', 'R', 'F']
    if metric_type not in metric_types:
        raise ValueError("metric_type must be one of {}".format(metric_types))
    metric2lbl_fn = dict(LBL_FNS)
    lbl_fn = (metric_type, metric2lbl_fn[metric_type])

    return parseval_similarity(parser_preds,
                               exclude_root=exclude_root,
                               subtree_filter=subtree_filter,
                               lbl_fn=lbl_fn,
                               span_type=span_type,
                               digits=digits,
                               percent=percent,
                               print_support=print_support,
                               per_doc=per_doc,
                               add_trivial_spans=add_trivial_spans,
                               out_format=out_format)


def rst_parseval_report(ctree_true, ctree_pred, ctree_type='RST',
                        subtree_filter=None, metric_types=None,
                        span_type='edus',
                        digits=4, percent=False, print_support_pred=True,
                        per_doc=False,
                        add_trivial_spans=False,
                        stringent=False):
    """Build a text report showing the PARSEVAL discourse metrics.

    This is the simplest report we need to generate, it corresponds
    to the arrays of results from the literature.
    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    ctree_true: TODO
        TODO

    ctree_pred: TODO
        TODO

    ctree_type : one of {'RST', 'SimpleRST'}, defaults to 'RST'
        Type of ctrees considered in the evaluation procedure.
        'RST' is the standard type of ctrees used in the RST corpus,
        it triggers the exclusion of the root node from the evaluation
        but leaves are kept.
        'SimpleRST' is a binarized variant of RST trees where each
        internal node corresponds to an attachment decision ; in other
        words, it is a binary ctree where the nuclearity and relation label
        are moved one node up compared to the standard RST trees. This
        triggers the exclusion of leaves from the eval, but the root node
        is kept.

    subtree_filter: function, optional
        Function to filter all local trees.

    metric_types : list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'N', 'R', 'F'].

    digits : int, defaults to 4
        Number of decimals to print.

    print_support_pred : boolean, defaults to True
        If True, the predicted support, i.e. the number of predicted
        spans, is also displayed. This is useful for non-binary ctrees
        as the number of spans in _true and _pred can differ.

    per_doc : boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.

    add_trivial_spans : boolean, defaults to False
        If True, trivial spans 0-0, 0-n, 1-n are added ; this is meant to
        replicate the evaluation procedure of Li et al.'s dependency RST
        parser.

    stringent : boolean, defaults to False
        TODO
    """
    # filter root or leaves, depending on the type of ctree
    if ctree_type not in ['RST', 'SimpleRST']:
        raise ValueError("ctree_type should be one of {'RST', 'SimpleRST'}")
    if ctree_type == 'RST':
        # standard RST ctree: exclude root
        exclude_root = True
        subtree_filter = subtree_filter
    elif ctree_type == 'SimpleRST':
        # SimpleRST variant: keep root, exclude leaves
        exclude_root = False  # TODO try True first, should get same as before
        not_leaf = lambda t: t.height() > 2  # TODO unit test!
        if subtree_filter is None:
            subtree_filter = not_leaf
        else:
            subtree_filter = lambda t: not_leaf(t) and subtree_filter(t)

    # select metrics and the corresponding functions
    if metric_types is None:
        # metric_types = ['S', 'N', 'R', 'F']
        metric_types = [x[0] for x in LBL_FNS]
    if set(metric_types) - set(x[0] for x in LBL_FNS):
        raise ValueError('Unknown metric types in {}'.format(metric_types))
    metric2lbl_fn = dict(LBL_FNS)
    lbl_fns = [(metric_type, metric2lbl_fn[metric_type])
               for metric_type in metric_types]

    return parseval_report(ctree_true, ctree_pred, exclude_root=exclude_root,
                           subtree_filter=subtree_filter, lbl_fns=lbl_fns,
                           span_type=span_type,
                           digits=digits,
                           percent=percent,
                           print_support_pred=print_support_pred,
                           per_doc=per_doc,
                           add_trivial_spans=add_trivial_spans)


def rst_parseval_detailed_report(ctree_true, ctree_pred, ctree_type='RST',
                                 subtree_filter=None, span_type='edus',
                                 metric_type='R',
                                 labels=None, sort_by_support=True,
                                 digits=4, percent=False, per_doc=False):
    """Build a text report showing the PARSEVAL discourse metrics per label.

    Metrics are calculated globally (average='micro').

    Parameters
    ----------
    ctree_true: TODO
        TODO

    ctree_pred: TODO
        TODO

    ctree_type : one of {'RST', 'SimpleRST'}, defaults to 'RST'
        Type of ctrees considered in the evaluation procedure.
        'RST' is the standard type of ctrees used in the RST corpus,
        it triggers the exclusion of the root node from the evaluation
        but leaves are kept.
        'SimpleRST' is a binarized variant of RST trees where each
        internal node corresponds to an attachment decision ; in other
        words, it is a binary ctree where the nuclearity and relation label
        are moved one node up compared to the standard RST trees. This
        triggers the exclusion of leaves from the eval, but the root node
        is kept.

    subtree_filter: function, optional
        Function to filter all local trees.

    metric_type : one of {'R', 'F'}, defaults to 'R'
        Metric that need to be included in the report.

    digits : int, defaults to 4
        Number of decimals to print.

    per_doc: boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.

    """
    # filter root or leaves, depending on the type of ctree
    if ctree_type not in ['RST', 'SimpleRST']:
        raise ValueError("ctree_type should be one of {'RST', 'SimpleRST'}")
    if ctree_type == 'RST':
        # standard RST ctree: exclude root
        exclude_root = True
        subtree_filter = subtree_filter
    elif ctree_type == 'SimpleRST':
        # SimpleRST variant: keep root, exclude leaves
        exclude_root = False  # TODO try True first, should get same as before
        not_leaf = lambda t: t.height() > 2  # TODO unit test!
        if subtree_filter is None:
            subtree_filter = not_leaf
        else:
            subtree_filter = lambda t: not_leaf(t) and subtree_filter(t)

    # select metrics and the corresponding functions
    if metric_type not in set(x[0] for x in LBL_FNS):
        raise ValueError('Unknown metric type: {}'.format(metric_type))
    metric2lbl_fn = dict(LBL_FNS)
    lbl_fn = (metric_type, metric2lbl_fn[metric_type])

    return parseval_detailed_report(
        ctree_true, ctree_pred, exclude_root=exclude_root,
        subtree_filter=subtree_filter, span_type=span_type,
        lbl_fn=lbl_fn,
        labels=labels, sort_by_support=sort_by_support,
        digits=digits, percent=percent, per_doc=per_doc)
