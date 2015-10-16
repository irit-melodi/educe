"""Statistical description of the corpus: EDUs, CDUs, relations.

"""

from __future__ import print_function

import pandas as pd

from educe.stac.annotation import is_cdu, is_edu, is_relation_instance
from educe.stac.context import Context


def rel_feats(doc, ctx, anno, debug=False):
    """Get features for relations.

    Parameters
    ----------
    doc:
    ctx:
    anno:

    Returns
    -------
    res: dict
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
    # * nb_gaps: CDUs spans can be discontiguous
    # * gap_max_len: max len of a gap (in #EDUs)
    # * over_nb_turns: nb of turns this CDU (partly) spans over
    # * over_nb_tstars: nb of tstars this CDU (partly) spans over

    res = {
        'nb_members': nb_members,
        'nb_cdus_imm': nb_cdus_imm,
        'nb_cdus_tot': nb_cdus_tot,
        'max_lvl': max_lvl,
        'nb_edus_tot': nb_edus_tot,
    }

    return res


def create_dfs(corpus):
    """Create pandas DataFrames for the corpus.

    Returns
    -------
    cdus:
    rels:
    """
    cdu_data = []
    rel_data = []
    for file_id, doc in corpus.items():
        # common stuff: get general info (doc, subdoc, annotator)
        doc_name = file_id.doc
        subdoc_name = file_id.subdoc
        stage = file_id.stage
        annotator = file_id.annotator
        # context: yerk
        ctx = Context.for_edus(doc)
        #
        for anno in doc.annotations():
            common_cols = {
                'anno_id': anno._anno_id,  # TODO get another way (hidden attribute)
                'doc': doc_name,
                'subdoc': subdoc_name,
                'stage': stage,
                'annotator': annotator,
                'type': anno.type,  # ? maybe not
            }
            if is_cdu(anno):
                row = dict(common_cols.items() +
                           cdu_feats(anno).items())
                cdu_data.append(row)
            elif is_relation_instance(anno):
                row = dict(common_cols.items() +
                           rel_feats(doc, ctx, anno).items())
                rel_data.append(row)
    # create dataframes
    cdus = pd.DataFrame(data=cdu_data)
    rels = pd.DataFrame(data=rel_data)

    return cdus, rels
