# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Show number of EDUs, turns, etc
"""

from __future__ import print_function
from collections import defaultdict

import educe.stac

from ..args import\
    add_usual_input_args, add_usual_output_args,\
    read_corpus_with_unannotated
from ..doc import strip_fixme

NAME = 'count'

# we have an order on this, so no dict
SEGMENT_CATEGORIES = [("dialogue", educe.stac.is_dialogue),
                      ("turn", educe.stac.is_turn),
                      ("edu", educe.stac.is_edu)]


LINK_CATEGORIES = [("relation instance", educe.stac.is_relation_instance),
                   ("CDU", educe.stac.is_cdu)]


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
            pred, extract = pred_extract
            if pred(anno):
                counts[extract(anno)] += 1
    return counts


def summary(counts,
            doc_counts=None,
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
    if keys is None:
        keys = counts.keys()

    def info(k):
        "summary line for a given key"
        if doc_counts is None:
            return str(counts[k])
        else:
            dcounts = [doc_counts[d][k] for d in doc_counts]
            mean, median = rounded_mean_median(dcounts)
            return "%d (%d-%d per doc, mean %d, median %d)" %\
                   (counts[k], min(dcounts), max(dcounts),
                    mean, median)

    lines = ["%s: %s" % (k, info(k)) for k in keys]
    if total is not False:
        lines.append("TOTAL: %d" % sum(counts.values()))
    return "\n".join(lines)


def big_banner(string, width=60):
    """
    Convert a string into a large banner ::

       foo
       ========================================

    """
    return "\n".join([string, "=" * width, ""])


def small_banner(string):
    """
    Convert a string into a small banner ::

       foo
       ---
    """
    return "\n".join([string, "-" * len(string)])


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


def sectioned_summary(s_counts, total=None):
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
        lines.append(small_banner(section))
        lines.append(summary(s_counts[section], total=total))
        lines.append("")
    lines.append(small_banner("all together"))
    lines.append(summary(combined, total=total))
    return "\n".join(lines)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus_with_unannotated(args, verbose=True)
    unannotated_keys = [k for k in corpus if k.stage == "unannotated"]
    annotators = frozenset(k.annotator for k in corpus
                           if k.annotator is not None)
    unanno_counts = empty_counts()
    unanno_doc_counts = defaultdict(empty_counts)

    for kdoc in frozenset(k.doc for k in unannotated_keys):
        ksubdocs = frozenset(k.subdoc for k in unannotated_keys
                             if k.doc == kdoc)
        unanno_counts["doc"] += 1
        unanno_counts["subdoc"] += len(ksubdocs)

        # separate counts for each doc so that we can collect
        # min/max/mean/median etc
        unanno_doc_counts[kdoc]["subdoc"] += len(ksubdocs)
        for k in (k for k in unannotated_keys if k.doc == kdoc):
            count(corpus[k], dict(SEGMENT_CATEGORIES),
                  counts=unanno_doc_counts[kdoc])

    for k in unannotated_keys:
        count(corpus[k], dict(SEGMENT_CATEGORIES),
              counts=unanno_counts)

    anno_acts = {}
    anno_rlabels = {}
    anno_links = {}
    for annotator in annotators:
        units, discourse = anno_subcorpus(corpus, annotator)
        anno_acts[annotator] = empty_counts()
        for k in units:
            count(corpus[k], {},
                  counts=anno_acts[annotator],
                  pred_extract=(educe.stac.is_edu, hinted_type))
        anno_links[annotator] = empty_counts()
        anno_rlabels[annotator] = empty_counts()
        for k in discourse:
            count(corpus[k], dict(LINK_CATEGORIES),
                  counts=anno_links[annotator])
            count(corpus[k], {},
                  counts=anno_rlabels[annotator],
                  pred_extract=(educe.stac.is_relation_instance,
                                lambda x: x.type))

    keys = ["subdoc"] + [k for k, _ in SEGMENT_CATEGORIES]
    lines = [big_banner("Document structure"),
             summary(unanno_counts, keys=["doc"],
                     total=False),
             summary(unanno_counts,
                     doc_counts=unanno_doc_counts,
                     keys=keys, total=False),
             "",
             big_banner("Links"),
             sectioned_summary(anno_links, total=False),
             "",
             big_banner("Dialogue acts"),
             sectioned_summary(anno_acts),
             "",
             big_banner("Relation instances"),
             sectioned_summary(anno_rlabels)]
    print("\n".join(lines))
