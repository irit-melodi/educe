"""Statistical description of the corpus: EDUs, CDUs, relations.

"""

from __future__ import print_function

import pandas as pd
from tabulate import tabulate

from educe.stac.annotation import (is_cdu, is_dialogue, is_edu,
                                   is_relation_instance, is_turn,
                                   is_turn_star)
from educe.stac.context import Context


# cc'ed from count
SEGMENT_CATEGORIES = [("dialogue", is_dialogue),
                      ("turn star", is_turn_star),
                      ("turn", is_turn),
                      ("edu", is_edu)]


LINK_CATEGORIES = [("rel insts", is_relation_instance),
                   ("CDUs", is_cdu)]


def big_banner(string, width=60):
    """
    Convert a string into a large banner ::

       foo
       ========================================

    """
    return "\n".join([string, "=" * width, ""])
# end cc'ed


def rel_feats(doc, ctx, anno, debug=False):
    """Get features for relations.

    Parameters
    ----------
    doc : GlozzDocument
        Surrounding document

    ctx :

    anno :

    Returns
    -------
    res : dict(string, string?)
        Features for this relation
    """
    # get all EDUs from document, sorted by their span
    doc_edus = sorted([u for u in doc.units if is_edu(u)],
                      key=lambda u: u.span)
    # TODO doc_tstars = ...

    src = anno.source
    if is_cdu(src):
        src_type = 'CDU'
        src_edus = sorted(src.terminals(), key=lambda e: e.span)
    elif is_edu(src):
        src_type = 'EDU'
        src_edus = [src]
    else:
        # covered by stac-check ("non-DU endpoints")
        return {}

    tgt = anno.target
    if is_cdu(tgt):
        tgt_type = 'CDU'
        tgt_edus = sorted(tgt.terminals(), key=lambda e: e.span)
    elif is_edu(tgt):
        tgt_type = 'EDU'
        tgt_edus = [tgt]
    else:
        # covered by stac-check ("non-DU endpoints")
        return {}

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
        return {}

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
    src_tstars = [ctx[e].tstar for e in src_edus]
    tgt_tstars = [ctx[e].tstar for e in tgt_edus]
    interv_tstars = [ctx[e].tstar for e in interv_edus]
    # turn-stars from the interval that don't overlap with src nor tgt
    skipped_tstars = set(interv_tstars) - set(src_tstars) - set(tgt_tstars)
    # we define:
    # * tstar_dist = 0  if (part of) src and tgt belong to the same tstar
    # * tstar_dist = len(skipped_tstars) + 1 otherwise
    tstar_dist = (len(skipped_tstars) + 1
                  if not set(src_tstars).intersection(set(tgt_tstars))
                  else 0)

    res = {
        'src_type': src_type,
        'tgt_type': tgt_type,
        'direction': direction,
        'edu_dist': edu_dist,
        'tstar_dist': tstar_dist,
    }

    return res


def dlg_feats(anno):
    """Get Dialogue features."""
    span_char_len = anno.span.char_end - anno.span.char_start
    res = {
        'span_char_len': span_char_len,
    }
    # TODO maybe extend the dict with anno.features.items()
    # TODO span_tok_len: length in tokens (need info from CoreNLP)
    return res


def edu_feats(doc, ctx, anno):
    """Get EDU features.

    Parameters
    ----------
    doc : GlozzDocument
        Currently unused ; possibly there for compatibility with an
        attempted wider API (?).

    ctx : ?
        Some notion of context?

    anno : Unit
        An EDU, as an annotation.

    Returns
    -------
    res : dict(string, string?)
        Features for this EDU.
    """
    span_char_len = anno.span.char_end - anno.span.char_start
    turn = ctx[anno].turn.identifier()
    tstar = ctx[anno].tstar.identifier()
    dialogue = ctx[anno].dialogue.identifier()
    res = {
        'span_char_len': span_char_len,
        'turn': turn,
        'tstar': tstar,
        'dialogue': dialogue,
    }

    # TODO maybe extend the dict with anno.features.items()
    # TODO span_tok_len: length in tokens (need info from CoreNLP)
    return res


def turn_feats(anno):
    """Get Turn features."""
    span_char_len = anno.span.char_end - anno.span.char_start
    res = {
        'span_char_len': span_char_len,
    }
    # TODO maybe extend the dict with anno.features.items()
    # TODO span_tok_len: length in tokens (need info from CoreNLP)
    return res


def tstar_feats(anno):
    """Get turn-star features."""
    span_char_len = anno.span.char_end - anno.span.char_start
    res = {
        'span_char_len': span_char_len,
    }
    # TODO maybe extend the dict with anno.features.items()
    # TODO span_tok_len: length in tokens (need info from CoreNLP)
    return res


def cdu_feats(anno):
    """Get CDU features that are not immediate.

    Parameters
    ----------
    anno: Schema
        The schema that codes this CDU in the glozz format

    Returns
    -------
    res: dict(string, int)
        Features on this CDU, currently
        'nb_edus_tot' (total number of EDUs spanned by this CDU),
        'nb_cdus_imm' (number of CDUs immediately embedded in this CDU),
        'nb_cdus_tot' (total number of CDUs recursively embedded in this
        CDU),
        'max_lvl' (maximal degree of CDU nesting in this CDU).
    """
    nb_members = len(anno.members)
    nb_cdus_imm = len([m for m in anno.members if is_cdu(m)])

    nb_edus_tot = 0
    nb_cdus_tot = 0
    max_lvl = 0

    cdus_to_expand = [(0, anno)]
    while cdus_to_expand:
        lvl, cur_cdu = cdus_to_expand.pop()
        mem_lvl = lvl + 1
        for member in cur_cdu.members:
            if is_edu(member):
                nb_edus_tot += 1
            elif is_cdu(member):
                nb_cdus_tot += 1
                if mem_lvl > max_lvl:
                    max_lvl = mem_lvl
                cdus_to_expand.append((mem_lvl, member))
            else:
                raise ValueError('Unexpected type for a CDU member')

    # TODO new features:
    # * nb_gaps: CDUs spans can be discontinuous
    # * gap_max_len: max len of a gap (in #EDUs)
    # * over_nb_turns: nb of turns this CDU (partly) spans over
    # * over_nb_tstars: nb of tstars this CDU (partly) spans over

    res = {
        'members': nb_members,
        'members_cdu': nb_cdus_imm,
        'spanned_cdus': nb_cdus_tot,
        'spanned_edus': nb_edus_tot,
        'depth': max_lvl,
    }

    return res


def create_dfs(corpus):
    """Create pandas DataFrames for the corpus.

    Returns
    -------
    res: dict(string, DataFrame)
        A DataFrame for each kind of structure present in the corpus.
    """
    rows = {anno_type: list()
            for anno_type in ['edu', 'turn', 'tstar', 'dialogue',
                              'cdu', 'rel']}

    for file_id, doc in corpus.items():
        # common stuff: get general info (doc, subdoc, annotator)
        doc_name = file_id.doc
        subdoc_name = file_id.subdoc
        stage = file_id.stage
        annotator = file_id.annotator
        # context: yerk
        ctx = Context.for_edus(doc)
        # doc.annotations() := doc.units + doc.relations + doc.schemas
        for anno in doc.annotations():
            common_cols = {
                'anno_id': anno.identifier(),
                'doc': doc_name,
                'subdoc': subdoc_name,
                'stage': stage,
                'annotator': annotator,
                'type': anno.type,  # ? maybe not
            }
            if is_edu(anno):
                row = dict(common_cols.items() +
                           edu_feats(doc, ctx, anno).items())
                rows['edu'].append(row)
            elif is_cdu(anno):
                row = dict(common_cols.items() +
                           cdu_feats(anno).items())
                rows['cdu'].append(row)
            elif is_relation_instance(anno):
                row = dict(common_cols.items() +
                           rel_feats(doc, ctx, anno).items())
                rows['rel'].append(row)
            elif is_dialogue(anno):
                row = dict(common_cols.items() +
                           dlg_feats(anno).items())
                rows['dialogue'].append(row)
            elif is_turn(anno):
                row = dict(common_cols.items() +
                           turn_feats(anno).items())
                rows['turn'].append(row)
            elif is_turn_star(anno):
                row = dict(common_cols.items() +
                           tstar_feats(anno).items())
                rows['tstar'].append(row)
            elif anno.type in ['paragraph',
                               'Resource', 'Anaphora',
                               'Several_resources', 'Preference']:
                # each paragraph (normally) corresponds to a Turn
                # so just ignore them ;
                # the situation is less clear-cut for 'Resource',
                # 'Anaphora', 'Several_resources'
                continue
            else:
                err_msg = 'Unsupported annotation: {}'.format(anno)
                # raise ValueError(err_msg)
                print('W: {}'.format(err_msg))
                continue

    res = {anno_type: pd.DataFrame(data=row_list)
           for anno_type, row_list in rows.items()
           if row_list}

    return res


def report_on_corpus(corpus):
    """Prepare and print a report on this corpus."""
    # TODO create separate sets of DataFrames for "unannotated", "discourse"
    # and "units"
    dfs = create_dfs(corpus)
    # invariant info
    turns = dfs['turn']
    # FIXME make turn-stars an integral part of a document, so they always
    # exist...
    if 'tstar' in dfs:
        tstars = dfs['tstar']  # maybe not so invariant
    # pre-annotation (seldom readjusted)
    edus = dfs['edu']
    dialogues = dfs['dialogue']
    # annotation
    if 'rel' in dfs:
        rels = dfs['rel']
        disc_rels = rels[rels['stage'] == 'discourse']
    else:
        disc_rels = pd.DataFrame()
    # pd.util.testing.assert_frame_equal(rels, disc_rels)
    # FIXME this assertion fails for TEST: a document has
    # an "Elaboration" relation in units/SILVER (!?) ; this is
    # probably an error from the annotation process
    if 'cdu' in dfs:
        cdus = dfs['cdu']
        disc_cdus = cdus[cdus['stage'] == 'discourse']
    else:
        disc_cdus = pd.DataFrame()
    # pd.util.testing.assert_frame_equal(cdus, disc_cdus)
    # TODO maybe filter to keep only the EDUs with a dialogue act annotation?
    acts = edus[edus['stage'] == 'units']

    # invariant info
    ua_turns = turns[turns['stage'] == 'unannotated']
    if 'tstar' in dfs:
        ua_tstars = tstars[tstars['stage'] == 'unannotated']
    # pre-annotation should be invariant to the annotation process
    ua_edus = edus[edus['stage'] == 'unannotated']
    ua_dialogues = dialogues[dialogues['stage'] == 'unannotated']
    # discourse annotation
    ds_edus = edus[edus['stage'] == 'discourse']

    # Document structure
    # ==================
    # computed on the "unannotated" layer, more precisely its EDUs
    # (with their context)
    print(big_banner('Document structure'))

    # per document
    print('Per document')
    all_stats = []
    for col_name in ['subdoc', 'dialogue', 'tstar', 'turn', 'anno_id']:
        cn_df = ua_edus[['doc', col_name]].drop_duplicates()
        if col_name == 'anno_id':
            cn_df.rename(columns={'anno_id': 'edu'}, inplace=True)
        cn_stats = cn_df.groupby('doc').count().describe()
        cn_stats_s = cn_stats.unstack().unstack()
        cn_stats_s['total'] = cn_df.count()
        all_stats.append(cn_stats_s)
    print(pd.concat(all_stats))
    print()

    # per dialogue
    print('Per dialogue')
    all_stats = []
    for col_name in ['tstar', 'turn', 'anno_id']:
        cn_df = ua_edus[['dialogue', col_name]].drop_duplicates()
        if col_name == 'anno_id':
            cn_df.rename(columns={'anno_id': 'edu'}, inplace=True)
        cn_stats = cn_df.groupby('dialogue').count().describe()
        cn_stats_s = cn_stats.unstack().unstack()
        cn_stats_s['total'] = cn_df.count()
        all_stats.append(cn_stats_s)
    print(pd.concat(all_stats))
    print()

    # per dialogue (2+ EDUs)
    print('Per dialogue (2+ EDUs)')
    # filter ua_edus for dialogues with 2+ EDUs: group by dialogue,
    # then filter groups with only one entry
    ua_edus_2p = ua_edus.groupby('dialogue').filter(lambda x: len(x) > 1)
    # gather stats
    all_stats = []
    for col_name in ['tstar', 'turn', 'anno_id']:
        cn_df = ua_edus_2p[['dialogue', col_name]].drop_duplicates()
        if col_name == 'anno_id':
            cn_df.rename(columns={'anno_id': 'edu'}, inplace=True)
        cn_stats = cn_df.groupby('dialogue').count().describe()
        cn_stats_s = cn_stats.unstack().unstack()
        cn_stats_s['total'] = cn_df.count()
        all_stats.append(cn_stats_s)
    print(pd.concat(all_stats))
    print()

    # on the discourse layer
    # per annotator
    headers_ator = ["annotator", "doc", "subdoc", "dialogue",
                    "turn star", "turn", "edu"]
    rows_ator = [[name,
                  group['doc'].nunique(),
                  group.drop_duplicates(['doc', 'subdoc'])['subdoc'].count(),
                  group['dialogue'].nunique(),
                  group['tstar'].nunique(),
                  group['turn'].nunique(),
                  group['anno_id'].nunique()]
                 for name, group in ds_edus.groupby('annotator')]
    rows_ator.append([
        "all together",
        ds_edus['doc'].nunique(),
        ds_edus.drop_duplicates(['doc', 'subdoc'])['subdoc'].count(),
        ds_edus['dialogue'].nunique(),
        ds_edus['tstar'].nunique(),
        ds_edus['turn'].nunique(),
        ds_edus['anno_id'].nunique(),
    ])
    # print these tables
    print(tabulate(rows_ator, headers=headers_ator))
    print()

    # Dialogue acts
    # =============
    if not acts.empty:
        print(big_banner("Dialogue acts"))

        acts_annot = acts.groupby('annotator')['type']
        print(acts_annot.count())
        print()
        print(acts_annot.value_counts())
        print()

    # Links (per annotator)
    # =====================
    if not disc_cdus.empty or not disc_rels.empty:
        print(big_banner('Links'))

    # CDUs
    if not disc_cdus.empty:
        print('CDUs: number of members')
        print('-----------------------')
        print(disc_cdus['members'].describe().to_frame().unstack().unstack())
        print(disc_cdus.groupby('annotator')['members'].describe().unstack())
        print()
    # rels
    if not disc_rels.empty:
        print('Relation instances: length (in EDUs)')
        print('------------------------------------')
        print(disc_rels['edu_dist'].describe().to_frame().unstack().unstack())
        print(disc_rels.groupby('annotator')['edu_dist'].describe().unstack())
        print()

    # Relation instances
    # ==================
    if not disc_rels.empty:
        print(big_banner("Relation instances"))

        print('Distribution of relations')
        print(disc_rels['type'].value_counts())
        print()
        print(disc_rels.groupby(['annotator'])['type'].value_counts())
        print()

        # additional tables
        print('Relation length (in EDUs)')
        rel_edist = (disc_rels.groupby('type')['edu_dist'].describe()
                     .unstack())
        print(rel_edist.sort_values(by='count', ascending=False))
        print()

        print('Relation length (in Turn-stars)')
        rel_tdist = (disc_rels.groupby('type')['tstar_dist'].describe()
                     .unstack())
        print(rel_tdist.sort_values(by='count', ascending=False))
        print()

        print('Type of endpoints')
        rel_by_endpoints = disc_rels.groupby(['src_type', 'tgt_type'])
        rel_by_endpoints_stats = rel_by_endpoints['edu_dist'].describe()
        print(rel_by_endpoints_stats.unstack()['count'].to_frame())
        print()
        if False:
            print('Type of endpoints, by relation')
            print(rel_by_endpoints['type'].value_counts().to_frame())
            print()

    # CDUs
    # ====
    if not disc_cdus.empty:
        print(big_banner("CDUs"))
        print(disc_cdus.describe().transpose())
