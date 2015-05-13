# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Show number of EDUs, turns, etc
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
import copy

from tabulate import tabulate

from ..args import (add_usual_input_args,
                    read_corpus_with_unannotated)
from ..doc import strip_fixme
from educe.util import concat
import educe.stac

# we have an order on this, so no dict
SEGMENT_CATEGORIES = [("dialogue", educe.stac.is_dialogue),
                      ("turn", educe.stac.is_turn),
                      ("edu", educe.stac.is_edu)]


LINK_CATEGORIES = [("rel insts", educe.stac.is_relation_instance),
                   ("CDUs", educe.stac.is_cdu)]


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    parser.set_defaults(func=main)


def rounded_mean_median(things):
    """
    Done in a fairly naive way
    In Python 3 we should just use statistics
    """
    length = len(things)
    middle = length / 2
    sorted_things = sorted(things)
    median = sorted_things[middle] if length % 2 else\
        (sorted_things[middle] + sorted_things[middle - 1]) / 2.0
    mean = float(sum(things)) / length
    return int(round(mean)), int(round(median))


def empty_counts():
    "A fresh set of counts"
    return defaultdict(int)


def count(doc,
          categories,
          counts=None,
          pred_extract=None):
    """
    Count the number of annotations in a document grouped into categories.
    An annotation may belong in more than one category. Returns a dictionary
    of textual keys to counts. If you supply a dictionary, the counts will
    be incremented there.

    :param categories: what to look for and how (uses a global default
                       otherwise)
    :type categories: dictionary of string to predicate

    :param pred_extract: in addition to categories, this is a single
                      predicate/extractor pairs, such that the
                      extractor assigns a category to any
                      annotation filtered in by the predicate
                      (hint: if you only want to use extractor,
                      just pass in categories={}) (empty by default)
    :type pred_extract: (anno -> bool, anno -> string)

    :param counts: accumulator if you want it
    :type extractors: dictionary of string to int
    """
    if counts is None:
        counts = defaultdict(int)
    for anno in doc.annotations():
        for cat, pred in categories.items():
            if pred(anno):
                counts[cat] += 1
        if pred_extract is not None:
            pred, extract = tuple(pred_extract)
            if pred(anno):
                counts[extract(anno)] += 1
    return counts


def summary(counts,
            doc_counts=None,
            title=None,
            keys=None,
            total=None):
    """
    (Multi-line) string summary of a categories dict.

    doc_counts gives per-document stats from which we can
    extract helpful details like means and medians

    If you supply the keys sequence, we use it both to select
    a subset of the keys and to assign an order to them.

    Total can be set to True/False depending on whether you
    want a final line for a total. If you set it to None,
    we use the default (true)
    """
    doc_counts = doc_counts or {}
    if keys is None:
        keys = counts.keys()

    dcount_keys = frozenset(concat(d.keys() for d in doc_counts.values()))
    has_doc_counts = any(k in dcount_keys for k in keys)
    rows = []
    for key in keys:
        row = [key, counts[key]]
        if key in dcount_keys:
            dcounts = [doc_counts[d][key] for d in doc_counts]
            mean, median = rounded_mean_median(dcounts)
            row += [min(dcounts),
                    max(dcounts),
                    mean,
                    median]
        elif has_doc_counts:
            row += [None, None, None, None]
        rows.append(row)
    if total is not False:
        rows.append(["TOTAL", sum(counts.values())])
        if has_doc_counts:
            row += [None, None, None, None]

    headers = [title or "", "total"]
    if has_doc_counts:
        headers += ["min", "max", "mean", "median"]
    return tabulate(rows, headers=headers)


def wide_summary(s_counts, keys=None):
    """
    Return a table of relation instance and CDU counts for each
    section
    """
    rows = []
    total = defaultdict(int)
    keys = keys or list(frozenset(concat(d.keys() for d in s_counts.values())))
    for section in s_counts:
        row = [section]
        for skey in keys:
            row.append(s_counts[section][skey])
            total[skey] += s_counts[section][skey]
        rows.append(row)
    rows.append(["all together"] + [total[x] for x in keys])
    headers = ["annotator"] + keys
    return tabulate(rows, headers=headers)


def big_banner(string, width=60):
    """
    Convert a string into a large banner ::

       foo
       ========================================

    """
    return "\n".join([string, "=" * width, ""])


def anno_subcorpus(corpus, annotator):
    """
    Return a tuple of sets of keys, first for units, second for
    discourse
    """
    by_anno = frozenset(k for k in corpus if k.annotator == annotator)
    units = frozenset(k for k in by_anno if k.stage == "units")
    discourse = frozenset(k for k in by_anno if k.stage == "discourse")
    return (units, discourse)


def hinted_type(anno):
    """
    Type annotation
    """

    def tidy(types):
        "minor touchups"
        return frozenset(strip_fixme(typ) for typ in types)

    def squish(types):
        """
        string representation for type set (should be singleton,
        but there are stragglers)
        """
        squished = "/".join(sorted(types))
        whitelist = {"Other/Strategic_comment": "Strategic_comment"}
        if len(types) == 1:
            return list(types)[0]
        elif squished in whitelist:
            return whitelist[squished]
        else:
            return squished + " (OBSOLETE: multi-type)"

    def rewrite(typ):
        "small hint message for weird types"
        if typ == "Strategic_comment":
            return typ + " (OBSOLETE => Other?)"
        elif typ == "Segment":
            return typ + " (unannotated)"
        else:
            return typ

    return rewrite(squish(tidy(educe.stac.split_type(anno))))


def tall_summary(s_counts, total=None):
    """
    More elaborate version of summary in which we have a two layer
    dict with `section -> key -> int`
    """
    combined = empty_counts()
    for section in s_counts:
        for key, val in s_counts[section].items():
            combined[key] += val
    lines = []
    for section in s_counts:
        lines.append(summary(s_counts[section],
                             total=total,
                             title=section))
        lines.append("")
    lines.append(summary(combined,
                         total=total,
                         title="all together"))
    return "\n".join(lines)


PerDoc = namedtuple("PerDoc",
                    ["total", "struct"])

PerAnno = namedtuple("PerAnno",
                     ["struct", "acts", "rlabels", "links"])

PerDialogue = namedtuple("PerDialogue",
                         ["total", "struct"])


def count_by_docname(corpus):
    """
    Return variety of counts by

    * document name
    * dialogue
    * dialogue with more than one edu
    """
    def count_segments(doc, output):
        "do segment counts; minor sugar"
        count(doc, dict(SEGMENT_CATEGORIES),
              counts=output)

    unannotated_keys = [k for k in corpus if k.stage == "unannotated"]
    dcounts = PerDoc(total=empty_counts(),
                     struct=defaultdict(empty_counts))
    gcounts = PerDialogue(total=empty_counts(),
                          struct=defaultdict(empty_counts))
    gcounts2 = PerDialogue(total=empty_counts(),
                           struct=defaultdict(empty_counts))

    for kdoc in frozenset(k.doc for k in unannotated_keys):
        ksubdocs = frozenset(k.subdoc for k in unannotated_keys
                             if k.doc == kdoc)
        dcounts.total["doc"] += 1
        dcounts.total["subdoc"] += len(ksubdocs)
        # separate counts for each doc so that we can collect
        # min/max/mean/median etc
        dcounts.struct[kdoc]["subdoc"] += len(ksubdocs)
        for k in (k for k in unannotated_keys if k.doc == kdoc):
            doc = corpus[k]
            count_segments(doc, dcounts.struct[kdoc])
            for dlg in doc.units:
                if not educe.stac.is_dialogue(dlg):
                    continue
                gdoc = copy.copy(doc)
                gdoc.units = [x for x in doc.units if dlg.encloses(x)]
                count_segments(gdoc, gcounts.struct[dlg])
                count_segments(gdoc, gcounts.total)
                if len([x for x in gdoc.units if educe.stac.is_edu(x)]) < 2:
                    continue
                # dialogues with more than one EDU
                count_segments(gdoc, gcounts2.struct[dlg])
                count_segments(gdoc, gcounts2.total)

    for k in unannotated_keys:
        count(corpus[k], dict(SEGMENT_CATEGORIES),
              counts=dcounts.total)
    return dcounts, gcounts, gcounts2


def count_by_annotator(corpus):
    """
    Return variety of by-annotator counts
    """
    annotators = frozenset(k.annotator for k in corpus
                           if k.annotator is not None)
    acounts = PerAnno(struct=defaultdict(empty_counts),
                      acts=defaultdict(empty_counts),
                      rlabels=defaultdict(empty_counts),
                      links=defaultdict(empty_counts))

    for annotator in annotators:
        units, discourse = anno_subcorpus(corpus, annotator)
        for kdoc in frozenset(k.doc for k in discourse):
            ksubdocs = frozenset(k.subdoc for k in discourse
                                 if k.doc == kdoc)
            acounts.struct[annotator]["doc"] += 1
            acounts.struct[annotator]["subdoc"] += len(ksubdocs)
        for k in units:
            count(corpus[k], {},
                  counts=acounts.acts[annotator],
                  pred_extract=(educe.stac.is_edu, hinted_type))
        for k in discourse:
            count(corpus[k], dict(SEGMENT_CATEGORIES),
                  counts=acounts.struct[annotator])
            count(corpus[k], dict(LINK_CATEGORIES),
                  counts=acounts.links[annotator])
            count(corpus[k], {},
                  counts=acounts.rlabels[annotator],
                  pred_extract=(educe.stac.is_relation_instance,
                                lambda x: x.type))
    return acounts


def report(dcounts, gcounts, gcounts2, acounts):
    """
    Return a full report of all our counts
    """
    keys = ["doc", "subdoc"] + [k for k, _ in SEGMENT_CATEGORIES]
    lines = [big_banner("Document structure"),
             summary(dcounts.total,
                     title="per doc",
                     doc_counts=dcounts.struct,
                     keys=keys,
                     total=False),
             "",
             summary(gcounts.total,
                     title="per dialogue",
                     doc_counts=gcounts.struct,
                     keys=[k for k, _ in SEGMENT_CATEGORIES],
                     total=False),
             "",
             summary(gcounts2.total,
                     title="per dlg (2+ EDUs)",
                     doc_counts=gcounts2.struct,
                     keys=[k for k, _ in SEGMENT_CATEGORIES],
                     total=False),
             "",
             wide_summary(acounts.struct,
                          keys=keys),
             "",
             big_banner("Links"),
             wide_summary(acounts.links),
             "",
             big_banner("Dialogue acts"),
             tall_summary(acounts.acts),
             "",
             big_banner("Relation instances"),
             tall_summary(acounts.rlabels)]
    return "\n".join(lines)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus_with_unannotated(args, verbose=True)
    dcounts, gcounts, gcounts2 = count_by_docname(corpus)
    acounts = count_by_annotator(corpus)
    print(report(dcounts, gcounts, gcounts2, acounts))
