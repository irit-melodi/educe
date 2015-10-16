# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Show number of EDUs, turns, etc
"""

from __future__ import print_function
from collections import Counter, defaultdict, namedtuple
import copy
import itertools

from tabulate import tabulate

from ..args import (add_usual_input_args,
                    read_corpus_with_unannotated)
from ..doc import strip_fixme
from .pd_count import create_dfs
from educe.stac.context import (Context, merge_turn_stars)
from educe.util import concat
import educe.stac
# from educe.stac.sanity.common import is_default


# we have an order on this, so no dict
SEGMENT_CATEGORIES = [("dialogue", educe.stac.is_dialogue),
                      ("turn star", lambda x: x.type == 'Tstar'),
                      ("turn", educe.stac.is_turn),
                      # temporary workaround to exclude Tstars from being
                      # considered as EDUs
                      # TODO update educe.stac.{annotation,context} to
                      # cleanly integrate Tstars
                      # MM: I cannot make an informed decision about this yet
                      ("edu", lambda x: (educe.stac.is_edu(x) and
                                         x.type != 'Tstar'))]


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
            # add Turn-stars to doc
            # TODO encapsulate upstream
            tstar_doc = merge_turn_stars(doc)
            for anno in tstar_doc.units:
                if educe.stac.is_turn(anno):
                    anno.type = 'Tstar'
                    doc.units.append(anno)
            # end add Turn-stars
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


# EXPERIMENTAL: CDU and relation stuff
def rel_feats(doc, ctxt, anno, debug=False):
    """Get features for relations.

    Parameters
    ----------
    doc:
    ctxt:
    anno:

    Returns
    -------
    src_type: string
        Type of the source discourse unit

    tgt_type: string
        Type of the target discourse unit

    direction: string
        Direction of the relation, i.e. left or right attachment

    edu_dist: int
        Distance in EDUs between source and target units.

    tstar_dist: int
        Distance in Turn-stars between source and target units.
    """
    # get all EDUs from document, sorted by their span
    doc_edus = sorted([u for u in doc.units if educe.stac.is_edu(u)],
                      key=lambda u: u.span)
    # TODO doc_tstars = ...

    src = anno.source
    if educe.stac.is_cdu(src):
        src_type = 'CDU'
        src_edus = sorted(src.terminals(), key=lambda e: e.span)
    elif educe.stac.is_edu(src):
        src_type = 'EDU'
        src_edus = [src]
    else:
        # covered by stac-check ("non-DU endpoints")
        return []

    tgt = anno.target
    if educe.stac.is_cdu(tgt):
        tgt_type = 'CDU'
        tgt_edus = sorted(tgt.terminals(), key=lambda e: e.span)
    elif educe.stac.is_edu(tgt):
        tgt_type = 'EDU'
        tgt_edus = [tgt]
    else:
        # covered by stac-check ("non-DU endpoints")
        return []

    # get the index of the EDUs in the interval between src and tgt
    src_idc = [doc_edus.index(e) for e in src_edus]
    tgt_idc = [doc_edus.index(e) for e in tgt_edus]

    # error case covered at least partially by stac-check, either
    # as "bizarre relation instance" or as "CDU punctures"
    if set(src_idc).intersection(set(tgt_idc)):
        if debug:
            direction = 'messed up'
            print('* {}: {} {}'.format(doc.origin, direction, anno.type))
            print('\t' + ', '.join(['[{}] {}'.format(str(e.span),
                                                     doc.text(e.span))
                                    for e in src_edus]))
            print('\t' + ', '.join(['[{}] {}'.format(str(e.span),
                                                     doc.text(e.span))
                                    for e in tgt_edus]))
        return []

    # src ... tgt
    if src_idc[-1] < tgt_idc[0]:
        direction = 'right'
        fst_idc = src_idc
        snd_idc = tgt_idc
        interv_edus = doc_edus[(fst_idc[-1] + 1):snd_idc[0]]
    # tgt ... src
    elif tgt_idc[-1] < src_idc[0]:
        direction = 'left'
        fst_idc = tgt_idc
        snd_idc = src_idc
        interv_edus = doc_edus[(fst_idc[-1] + 1):snd_idc[0]]
    # tgt and src are interwoven
    else:
        direction = 'interwoven'  # FIXME
        src_tgt_idc = set(src_idc).union(tgt_idc)
        interv_edus = []
        gap_edus = [e for i, e in enumerate(doc_edus)
                    if (i not in src_tgt_idc and
                        i > min(src_tgt_idc) and
                        i < max(src_tgt_idc))]
        if debug:
            print('* {}: {} {}'.format(doc.origin, direction, anno.type))
            print('\t' + ', '.join(['[{}] {}'.format(str(e.span),
                                                     doc.text(e.span))
                                    for e in src_edus]))
            print('\t' + ', '.join(['[{}] {}'.format(str(e.span),
                                                     doc.text(e.span))
                                    for e in tgt_edus]))
            print('\t' + ', '.join(['[{}] {}'.format(str(e.span),
                                                     doc.text(e.span))
                                    for e in gap_edus]))
    edu_dist = len(interv_edus) + 1

    # turn-stars distance
    src_tstars = [ctxt[e].tstar for e in src_edus]
    tgt_tstars = [ctxt[e].tstar for e in tgt_edus]
    interv_tstars = [ctxt[e].tstar for e in interv_edus]
    # turn-stars from the interval that don't overlap with src nor tgt
    skipped_tstars = set(interv_tstars) - set(src_tstars) - set(tgt_tstars)
    # we define:
    # * tstar_dist = 0  if (part of) src and tgt belong to the same tstar
    # * tstar_dist = len(skipped_tstars) + 1 otherwise
    tstar_dist = (len(skipped_tstars) + 1
                  if not set(src_tstars).intersection(set(tgt_tstars))
                  else 0)

    return src_type, tgt_type, direction, edu_dist, tstar_dist


def cdu_feats(anno):
    """Get CDU features that are not immediate.

    Returns
    -------
    nb_edus_tot: int
        Total number of EDUs spanned by this CDU.

    nb_cdus_imm: int
        Number of CDUs immediately embedded in this CDU.

    nb_cdus_tot: int
        Total number of CDUs recursively embedded in this CDU.

    max_lvl: int
        Maximal degree of CDU nesting in this CDU.
    """
    nb_members = len(anno.members)
    nb_cdus_imm = len([m for m in anno.members
                       if educe.stac.is_cdu(m)])

    nb_edus_tot = 0
    nb_cdus_tot = 0
    max_lvl = 0

    cdus_to_expand = [(0, anno)]
    while cdus_to_expand:
        lvl, cur_cdu = cdus_to_expand.pop()
        mem_lvl = lvl + 1
        for member in cur_cdu.members:
            if educe.stac.is_edu(member):
                nb_edus_tot += 1
            elif educe.stac.is_cdu(member):
                nb_cdus_tot += 1
                if mem_lvl > max_lvl:
                    max_lvl = mem_lvl
                cdus_to_expand.append((mem_lvl, member))
            else:
                raise ValueError('Unexpected type for a CDU member')

    # TODO new features:
    # * nb_gaps: CDUs spans can be discontiguous
    # * gap_max_len: max len of a gap (in #EDUs)
    # * over_nb_turns: nb of turns this CDU (partly) spans over
    # * over_nb_tstars: nb of tstars this CDU (partly) spans over

    return nb_members, nb_cdus_imm, nb_cdus_tot, max_lvl, nb_edus_tot


def cdu_rel_statistics(corpus):
    """
    Return statistics on CDUs and relations
    """
    annotators = frozenset(k.annotator for k in corpus
                           if k.annotator is not None)

    cdus = []
    rels = []
    for annotator in annotators:
        units, discourse = anno_subcorpus(corpus, annotator)
        for k in discourse:
            doc = corpus[k]
            # TODO refactor Context to not be a dict(EDU, ...)
            ctxt = Context.for_edus(doc)
            # common "meta" features
            doc_name = doc.origin.doc
            subdoc_name = doc.origin.subdoc

            for anno in doc.annotations():
                row = [
                    # unique identifier
                    anno._anno_id,
                    # situation
                    doc_name, subdoc_name, annotator,
                    # type of annotation
                    anno.type,
                ]
                # extract feats for CDUs and rels
                if educe.stac.is_cdu(anno):
                    feats = cdu_feats(anno)
                    if feats:
                        row.extend(feats)
                        cdus.append(tuple(row))
                elif educe.stac.is_relation_instance(anno):
                    feats = rel_feats(doc, ctxt, anno)
                    if feats:
                        row.extend(feats)
                        rels.append(tuple(row))
    return cdus, rels


def cdu_report(cdu_stats):
    """Generate a string that contains reports on CDUs"""
    # detailed info on CDUs
    headers = ['CDUs', 'min', 'max', 'mean', 'median']
    rows = []
    # EDUs
    nb_edus_tot = [cs[-1] for cs in cdu_stats]
    mean_nb_edus_tot, median_nb_edus_tot = rounded_mean_median(nb_edus_tot)
    min_nb_edus_tot = min(nb_edus_tot)
    max_nb_edus_tot = max(nb_edus_tot)
    rows.append(['# EDUs',
                 min_nb_edus_tot, max_nb_edus_tot,
                 mean_nb_edus_tot, median_nb_edus_tot])
    # degree of nesting
    max_lvls = [cs[-2] for cs in cdu_stats]
    mean_lvl, median_lvl = rounded_mean_median(max_lvls)
    min_max_lvl = min(max_lvls)
    max_max_lvl = max(max_lvls)
    rows.append(['deg. nesting',
                 min_max_lvl, max_max_lvl,
                 mean_lvl, median_lvl])
    res = tabulate(rows, headers=headers)

    # additional info
    if True:
        # empty CDUs: call stac-oneoff clean-schemas
        empty_cdus = [cs for cs in cdu_stats
                      if cs[-1] == 0]
        if empty_cdus:
            print()
            print('Empty CDUs !?')
            print('\n'.join(str(cs)
                            for cs in sorted(empty_cdus,
                                             key=lambda c: (c[1], c[2]))))
            print()
        # CDUs with one member: call ???
        mono_member_cdus = [cs for cs in cdu_stats
                            if cs[-1] == 1]
        if mono_member_cdus:
            print()
            print('CDUs with a unique member !?')
            print('\n'.join(str(cs)
                            for cs in sorted(mono_member_cdus,
                                             key=lambda c: (c[1], c[2]))))
            print()
    if False:
        # CDUs occurring at the same level (nb_cdus_tot > max_lvl)
        same_lvl_cdus = [cs for cs in cdu_stats
                         if cs[-3] > cs[-2]]
        print(same_lvl_cdus)
        print()

    return res


def rel_report(rel_stats):
    """Generate a string that contains reports on relations"""
    # last fields: src_type, tgt_type, direction, edu_dist, tstar_dist

    # detailed info on relations
    headers = ['Relations', 'min', 'max', 'mean', 'median']
    rows = []
    # EDU dist
    dist_edus = [cs[-2] for cs in rel_stats]
    mean_dist_edus, median_dist_edus = rounded_mean_median(dist_edus)
    min_dist_edus = min(dist_edus)
    max_dist_edus = max(dist_edus)
    rows.append(['dist. EDUs',
                 min_dist_edus, max_dist_edus,
                 mean_dist_edus, median_dist_edus])
    # turn-star dist
    dist_tstars = [cs[-1] for cs in rel_stats]
    mean_dist_tstars, median_dist_tstars = rounded_mean_median(dist_tstars)
    min_dist_tstars = min(dist_tstars)
    max_dist_tstars = max(dist_tstars)
    rows.append(['dist. Turn-stars',
                 min_dist_tstars, max_dist_tstars,
                 mean_dist_tstars, median_dist_tstars])
    res = tabulate(rows, headers=headers)

    # (src_type, tgt_type)
    res += '\n\n'
    src_tgt_type_cnt = Counter([(cs[-5], cs[-4])
                                for cs in rel_stats])
    headers = ['(src_type, tgt_type)', '#occ.']
    rows = src_tgt_type_cnt.most_common()
    res += tabulate(rows, headers=headers)

    # TODO direction? + more detailed stats, possibly with pandas
    # TODO write an pandas-based replacement for this "count" util

    return res

# end EXPERIMENTAL: CDU and relation stuff


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
    if False:
        corpus = read_corpus_with_unannotated(args, verbose=True)
        dcounts, gcounts, gcounts2 = count_by_docname(corpus)
        acounts = count_by_annotator(corpus)
        print(report(dcounts, gcounts, gcounts2, acounts))

        # EXPERIMENTAL
        print()
        cdu_stats, rel_stats = cdu_rel_statistics(corpus)
        print(cdu_report(cdu_stats))
        print()
        print(rel_report(rel_stats))
        # end EXPERIMENTAL

    # EXPERIMENTAL-ER
    if True:
        corpus = read_corpus_with_unannotated(args, verbose=True)
        cdus, rels = create_dfs(corpus)

        print('\n'.join([
            big_banner('CDUs'),
            tabulate(cdus.describe(), headers="keys"),
            '',
        ]))

        print('\n'.join([
            big_banner('Relations'),
            tabulate(rels.describe(), headers="keys"),
            '',
        ]))

        # print distribution of relations
        # FIXME: new finds 1 SILVER/Elaboration in excess (79 vs 78)
        print(big_banner('Relation instances'))
        # for each annotator
        rels_by_annotator = rels.groupby('annotator')
        for name, group in rels_by_annotator:
            headers = [name, 'total']
            rows = group['type'].value_counts()
            # add total row to match the old version of "count"
            row_total = [('TOTAL', group['type'].count())]
            print(tabulate(list(rows.iteritems()) + row_total,
                           headers=headers))
            print()
        headers = ['all together', 'total']
        rows = rels['type'].value_counts()
        # add total row to match the old version of "count"
        row_total = [('TOTAL', rels['type'].count())]
        print(tabulate(list(rows.iteritems()) + row_total,
                       headers=headers))
    # end EXPERIMENTAL-ER
