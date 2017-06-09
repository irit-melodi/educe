"""Parseval metrics for constituency trees.

TODO
----
* [ ] factor out the report from the parseval scoring function, see
`sklearn.metrics.classification.classification_report`
* [ ] refactor the selection functions that enable to break down
evaluations, to avoid almost duplicates (as currently)
"""

from __future__ import absolute_import, print_function
import warnings

import numpy as np

from educe.metrics.scores_structured import (precision_recall_fscore_support,
                                             unique_labels)


def parseval_scores(ctree_true, ctree_pred, subtree_filter=None,
                    exclude_root=False, lbl_fn=None, labels=None,
                    span_type='edus',
                    average=None, per_doc=False,
                    add_trivial_spans=False):
    """Compute PARSEVAL scores for ctree_pred wrt ctree_true.

    Parameters
    ----------
    ctree_true : list of list of RSTTree or SimpleRstTree
        List of reference RST trees, one per document.

    ctree_pred : list of list of RSTTree or SimpleRstTree
        List of predicted RST trees, one per document.

    subtree_filter : function, optional
        Function to filter all local trees.

    exclude_root : boolean, defaults to True
        If True, exclude the root node of both ctrees from the eval.

    lbl_fn: function, optional
        Function to relabel spans.

    labels : list of string, optional
        Corresponds to sklearn's target_names IMO

    average : one of {'micro', 'macro'}, optional
        TODO, see scores_structured

    per_doc : boolean, optional
        If True, precision, recall and f1 are computed for each document
        separately then averaged over documents.
        (TODO this should probably be pushed down to
        `scores_structured.precision_recall_fscore_support`)

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Weighted average of the precision of each class.

    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support_true : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_true``.

    support_pred : int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``ctree_pred``.

    """
    # WIP
    if add_trivial_spans:
        # force inclusion of root span 1-n
        exclude_root = False

    # extract descriptions of spans from the true and pred trees
    spans_true = [ct.get_spans(subtree_filter=subtree_filter,
                               exclude_root=exclude_root,
                               span_type=span_type)
                  for ct in ctree_true]
    spans_pred = [ct.get_spans(subtree_filter=subtree_filter,
                               exclude_root=exclude_root,
                               span_type=span_type)
                  for ct in ctree_pred]

    # WIP replicate eval in Li et al.'s dep parser
    if add_trivial_spans:
        # add trivial spans for 0-0 and 0-n
        # this assumes n-n is the last span so we can get "n" as
        # sp_list[-1][0][1]
        spans_true = [sp_list + [((0, 0), "Root", '---', 0),
                                 ((0, sp_list[-1][0][1]), "Root", '---', 0)]
                      for sp_list in spans_true]
        spans_pred = [sp_list + [((0, 0), "Root", '---', 0),
                                 ((0, sp_list[-1][0][1]), "Root", '---', 0)]
                      for sp_list in spans_pred]
        # if label != span, change nuclearity to Satellite
        spans_true = [[(x[0], "Satellite" if x[2].lower() != "span" else x[1],
                        x[2], x[3]) for x in sp_list]
                      for sp_list in spans_true]
        spans_pred = [[(x[0], "Satellite" if x[2].lower() != "span" else x[1],
                        x[2], x[3]) for x in sp_list]
                      for sp_list in spans_pred]
    # end WIP
    # use lbl_fn to define labels
    if lbl_fn is not None:
        spans_true = [[(span[0], lbl_fn(span)) for span in spans]
                      for spans in spans_true]
        spans_pred = [[(span[0], lbl_fn(span)) for span in spans]
                      for spans in spans_pred]

    # NEW gather present labels
    present_labels = unique_labels(spans_true, spans_pred)
    if labels is None:
        labels = present_labels
    else:
        # currently not tested
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])
    # end NEW labels

    if per_doc:
        # non-standard variant that computes scores per doc then
        # averages them over docs ; this variant is implemented in DPLP
        # where it is mistaken for the standard version
        scores = []
        for doc_spans_true, doc_spans_pred in zip(spans_true, spans_pred):
            p, r, f1, s_true, s_pred = precision_recall_fscore_support(
                [doc_spans_true], [doc_spans_pred], labels=labels,
                average=average)
            scores.append((p, r, f1, s_true, s_pred))
        p, r, f1, s_true, s_pred = (
            np.array([x[0] for x in scores]).mean(),
            np.array([x[1] for x in scores]).mean(),
            np.array([x[2] for x in scores]).mean(),
            np.array([x[3] for x in scores]).sum(),
            np.array([x[4] for x in scores]).sum()
        )
    else:
        # standard version of this eval
        p, r, f1, s_true, s_pred = precision_recall_fscore_support(
            spans_true, spans_pred, labels=labels, average=average)

    return p, r, f1, s_true, s_pred, labels


def parseval_compact_report(parser_true, parser_preds,
                            exclude_root=False, subtree_filter=None,
                            lbl_fns=None,
                            span_type='edus',
                            digits=4,
                            percent=False,
                            print_support=True,
                            per_doc=False,
                            add_trivial_spans=False):
    """Build a text report showing the F1-scores of the PARSEVAL metrics
    for a list of parsers.

    This is the simplest and most compact report we need to generate, it
    corresponds to the comparative arrays of results from the literature.
    Metrics are calculated globally (average='micro'), unless per_doc is
    True (macro-averaging across documents).

    Parameters
    ----------
    parser_true: str
        Name of the parser used as reference ; it needs to be in the
        keys of parser_preds.

    parser_preds: list of (parser_name, ctree_pred)
        Predicted c-trees for each parser.

    metric_types: list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'N', 'R', 'F'].

    digits: int, defaults to 4
        Number of decimals to print.

    span_sel: TODO
        TODO

    per_doc: boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.
    """
    if lbl_fns is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fns = [('Labelled Span', lambda span_lbl: span_lbl[1])]

    metric_types = [k for k, v in lbl_fns]

    # prepare scaffold for report
    width = max(len(parser_name) for parser_name, _ in parser_preds)

    headers = [x for x in metric_types]
    if print_support:
        headers += ["support"]
    fmt = '%% %ds' % width  # first col: parser name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # display percentages
    if percent:
        digits = digits - 2

    # find _true
    for parser_name, ctree_pred in parser_preds:
        if parser_name == parser_true:
            ctree_true = ctree_pred
            break
    else:
        raise ValueError('Unable to find reference c-trees')

    for parser_name, ctree_pred in parser_preds:
        values = [parser_name]
        # compute scores
        metric_scores = dict()
        for metric_type, lbl_fn in lbl_fns:
            p, r, f1, s_true, s_pred, labels = parseval_scores(
                ctree_true, ctree_pred, subtree_filter=subtree_filter,
                exclude_root=exclude_root, lbl_fn=lbl_fn, labels=None,
                span_type=span_type,
                average='micro', per_doc=per_doc,
                add_trivial_spans=add_trivial_spans)
            metric_scores[metric_type] = (p, r, f1, s_true, s_pred)

        # fill report
        support = 0
        for metric_type in metric_types:
            (p, r, f1, s_true, s_pred) = metric_scores[metric_type]
            values += ["{0:0.{1}f}".format(f1 * 100.0 if percent else f1,
                                           digits)]
            # (warning) support in _true and _pred should be the same ;
            if s_true != s_pred:
                warnings.warn("s_pred != s_true")
            # store support in _true, for optional display below
            if support == 0:
                support = s_true
        # append support
        if print_support:
            values += ["{0:.0f}".format(support)]  # support_true

        report += fmt % tuple(values)

    return report


def parseval_similarity(parser_preds,
                        exclude_root=False, subtree_filter=None,
                        lbl_fn=None,
                        span_type='edus',
                        digits=4,
                        percent=False,
                        print_support=True,
                        per_doc=False,
                        add_trivial_spans=False,
                        out_format='str'):
    """Build a similarity matrix showing the F1-scores of a PARSEVAL metric
    for a list of parsers.

    Metrics are calculated globally (average='micro'), unless per_doc is
    True (macro-averaging across documents).

    Parameters
    ----------
    parser_preds : list of (parser_name, ctree_pred)
        Predicted c-trees for each parser.

    exclude_root : TODO
        TODO

    subtree_filter : TODO
        TODO

    lbl_fn : (str, function)
        Metric on which the similarity is computed.

    span_type : TODO
        TODO

    digits : int, defaults to 4
        Number of decimals to print.

    percent : TODO
        TODO

    print_support : TODO
        TODO

    per_doc : boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.

    add_trivial_spans : TODO
        TODO

    out_format : str, one of {'str', 'latex'}
        Output format.
    """
    if lbl_fn is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fn = ('Labelled Span', lambda span_lbl: span_lbl[1])

    metric_type = lbl_fn[0]

    # prepare scaffold for report
    width = max(len(parser_name) for parser_name, _ in parser_preds)
    headers = [k[:7] for k, v in parser_preds]
    if print_support:
        headers += ["support"]
    fmt = '%% %ds' % width  # first col: parser name
    if out_format == 'str':
        fmt += '  '
        fmt += ' '.join(['% 9s' for _ in headers])
    elif out_format == 'latex':
        fmt += ' &'
        fmt += '&'.join(['% 9s' for _ in headers])
        fmt += '\\\\'  # print "\\"
    else:
        raise ValueError("Unknown value for out_format: {}".format(
            out_format))
    fmt += '\n'
    headers = [""] + headers

    report = ""
    if out_format == 'latex':
        report += '\n'.join([
            '\\begin{table}[h]',
            '\\begin{center}',
            '\\begin{tabular}{' + 'l' * len(headers) +'}',
            '\\toprule',
            ''
        ])
    report += fmt % tuple(headers)
    report += '\n'
    if out_format == 'latex':
        report += '\\midrule\n'

    # display percentages
    if percent:
        digits = digits - 2

    for parser_true, ctree_true in parser_preds:
        values = [parser_true]
        for parser_name, ctree_pred in parser_preds:
            # compute scores
            p, r, f1, s_true, s_pred, labels = parseval_scores(
                ctree_true, ctree_pred, subtree_filter=subtree_filter,
                exclude_root=exclude_root, lbl_fn=lbl_fn[1], labels=None,
                span_type=span_type,
                average='micro', per_doc=per_doc,
                add_trivial_spans=add_trivial_spans)
            # fill report
            values += ["{0:0.{1}f}".format(f1 * 100.0 if percent else f1,
                                           digits)]
            # store support in _true, for optional display below
            support = s_true

        # append support
        if print_support:
            values += ["{0:.0f}".format(support)]  # support_true

        report += fmt % tuple(values)

    if out_format == 'latex':
        report += '\n'.join([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{center}',
            '\\caption{\\label{ctree-sim} Similarity matrix on parsers predictions against non-binarized trees.}',
            '\\end{table}'
        ])
    report = report.replace('_', ' ')

    return report


def parseval_report(ctree_true, ctree_pred, exclude_root=False,
                    subtree_filter=None, lbl_fns=None, span_type='edus',
                    digits=4, percent=False,
                    print_support_pred=True, per_doc=False,
                    add_trivial_spans=False):
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
    metric_types: list of strings, optional
        Metrics that need to be included in the report ; if None is
        given, defaults to ['S', 'N', 'R', 'F'].
    digits: int, defaults to 4
        Number of decimals to print.
    print_support_pred: boolean, defaults to True
        If True, the predicted support, i.e. the number of predicted
        spans, is also displayed. This is useful for non-binary ctrees
        as the number of spans in _true and _pred can differ.
    span_sel: TODO
        TODO
    per_doc: boolean, defaults to False
        If True, compute p, r, f for each doc separately then compute the
        mean of each score over docs. This is *not* the correct
        implementation, but it corresponds to that in DPLP.
    """
    if lbl_fns is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fns = [('Labelled Span', lambda span_lbl: span_lbl[1])]

    metric_types = [k for k, v in lbl_fns]

    # prepare scaffold for report
    width = max(len(str(x)) for x in metric_types)
    width = max(width, digits)
    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # display percentages
    if percent:
        digits = digits - 2

    # compute scores
    metric_scores = dict()
    for metric_type, lbl_fn in lbl_fns:
        p, r, f1, s_true, s_pred, labels = parseval_scores(
            ctree_true, ctree_pred, subtree_filter=subtree_filter,
            exclude_root=exclude_root, lbl_fn=lbl_fn, labels=None,
            span_type=span_type,
            average='micro', per_doc=per_doc,
            add_trivial_spans=add_trivial_spans)
        metric_scores[metric_type] = (p, r, f1, s_true, s_pred)

    # fill report
    if percent:
        digits = digits - 2
    for metric_type in metric_types:
        (p, r, f1, s_true, s_pred) = metric_scores[metric_type]
        values = [metric_type]
        for v in (p, r, f1):
            values += ["{0:0.{1}f}".format(v * 100.0 if percent else v,
                                           digits)]
        values += ["{0}".format(s_true)]  # support_true
        values += ["{0}".format(s_pred)]  # support_pred
        report += fmt % tuple(values)

    return report


def parseval_detailed_report(ctree_true, ctree_pred, exclude_root=False,
                             subtree_filter=None, lbl_fn=None,
                             span_type='edus',
                             labels=None, sort_by_support=True,
                             digits=4, percent=False, per_doc=False):
    """Build a text report showing the PARSEVAL discourse metrics.

    FIXME model after sklearn.metrics.classification.classification_report

    Parameters
    ----------
    ctree_true : list of RSTTree or SimpleRstTree
        Ground truth (correct) target structures.

    ctree_pred : list of RSTTree or SimpleRstTree
        Estimated target structures as predicted by a parser.

    labels : list of string, optional
        Relation labels to include in the evaluation.
        FIXME Corresponds more to target_names in sklearn IMHO.

    lbl_fn : function from tuple((int, int), (string, string)) to string
        Label extraction function

    digits : int
        Number of digits for formatting output floating point values.

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score, support for each
        class (or micro-averaged over all classes).

    """
    if lbl_fn is None:
        # we require a labelled span to be a pair (span, lbl)
        # where span and lbl can be anything, for example
        # * span = (span_beg, span_end)
        # * lbl = (nuc, rel)
        lbl_fn = ('Labelled Span', lambda span_lbl: span_lbl[1])
    # FIXME param lbl_fn is in fact a pair (metric_type, lbl_fn)
    metric_type, lbl_fn = lbl_fn

    # call with average=None to compute per-class scores, then
    # compute average here and print it
    p, r, f1, s_true, s_pred, labels = parseval_scores(
        ctree_true, ctree_pred, subtree_filter=subtree_filter,
        exclude_root=exclude_root, lbl_fn=lbl_fn, labels=labels,
        span_type=span_type,
        average=None, per_doc=per_doc)

    # scaffold for report
    last_line_heading = 'avg / total'

    width = max(len(str(lbl)) for lbl in labels)
    width = max(width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support", "sup_pred"]
    fmt = '%% %ds' % width  # first col: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # map labels to indices, possibly sorted by their support
    sorted_ilbls = enumerate(labels)
    if sort_by_support:
        sorted_ilbls = sorted(sorted_ilbls, key=lambda x: s_true[x[0]],
                              reverse=True)
    # display percentages
    if percent:
        digits = digits - 2
    # one line per label
    for i, label in sorted_ilbls:
        values = [label]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v * 100.0 if percent else v,
                                           digits)]
        values += ["{0}".format(s_true[i])]
        values += ["{0}".format(s_pred[i])]
        report += fmt % tuple(values)

    report += '\n'

    # last line ; compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s_true),
              np.average(r, weights=s_true),
              np.average(f1, weights=s_true)):
        values += ["{0:0.{1}f}".format(v * 100.0 if percent else v,
                                       digits)]
    values += ['{0}'.format(np.sum(s_true))]
    values += ['{0}'.format(np.sum(s_pred))]
    report += fmt % tuple(values)

    return report
