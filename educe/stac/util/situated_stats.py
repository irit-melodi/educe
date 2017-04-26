"""This module provides a growing library of functions to quantitatively
examine the situated version of the STAC corpus, in itself and with
respect to the purely linguistic (spect) version.
"""

from __future__ import absolute_import, print_function

from glob import glob
from itertools import chain
import os
import warnings

import pandas as pd

from educe.stac.annotation import (is_dialogue, is_edu, is_paragraph,
                                   is_preference, is_resource, is_turn)
from educe.stac.corpus import Reader as StacReader


# local path
STAC_DATA_DIR = '/home/mmorey/melodi/stac/svn-stac/data'

# naming schemes for games in each season
GAME_GLOBS = {
    'pilot': ['pilot*'],
    'socl-season1': ['s1-league*-game*'],
    'socl-season2': ['s2-league*-game*', 's2-practice*'],
}

# basic layout of the corpus: a few games from all seasons are explicitly set
# out for TEST, the rest are implicitly in TRAIN
SPLIT_GLOBS = {
    'TRAIN': [os.path.join(folder, file_glob)
              for folder, file_glob in chain(
                      (k, x) for k, v in GAME_GLOBS.items()
                      for x in v)],
    'TEST': [os.path.join('TEST', file_glob)
             for file_glob in set(chain.from_iterable(
                     GAME_GLOBS.values()))]
}

# base folder for each version
BASE_SPECT = ''
BASE_SITU = 'situated-annotation'

# TMP derogatory layout for _spect, should be fixed eventually
SPLIT_GLOBS_SPECT = {
    'TRAIN': [os.path.join(folder, file_glob)
              for folder, file_glob in chain(
                      (k + '_spect', x) for k, v in GAME_GLOBS.items()
                      for x in v)],
    'TEST': [os.path.join('TEST_spect', file_glob)
             for file_glob in set(chain.from_iterable(
                     GAME_GLOBS.values()))]
}

UNIT_COLS = [
    # identification
    'global_id',
    'doc',
    'subdoc',
    'stage',
    'annotator',
    # type, span, text
    'type',
    'span_beg',
    'span_end',
    'text',
    # metadata
    'creation_date',
    'author',
    'last_modif_date',  # optional?
    'last_modifier',  # optional?
]

TURN_COLS = UNIT_COLS + [
    'timestamp',
    'turn_id',
    'emitter',
    'developments',
    'resources',
    'comments',
]

SEG_COLS = UNIT_COLS

ACT_COLS = [
    'global_id',  # foreign key to SEG
    'surface_act',
    'addressee',
]

DLG_COLS = UNIT_COLS + [
    'gets',
    'trades',
    'dice_rolls',
]

RES_COLS = UNIT_COLS + [
    'status',
    'kind',
    'correctness',
    'quantity',
]

PREF_COLS = UNIT_COLS

SCHM_COLS = [
    # identification
    'global_id',
    'doc',
    'subdoc',
    'stage',
    'annotator',
    # type
    'type',
    # metadata
    'creation_date',
    'author',
    'last_modif_date',  # optional?
    'last_modifier',  # optional?
    # features?
    'operator',
    'default',
]

SCHM_MBRS_COLS = [
    'member_id',  # foreign key: global_id of schema or seg
    'schema_id',  # foreign key: global_id of schema
]

REL_COLS = [
    # identification
    'global_id',
    'doc',
    'subdoc',
    'stage',
    'annotator',
    # type
    'type',
    # metadata
    'creation_date',
    'author',
    'last_modif_date',  # optional?
    'last_modifier',  # optional?
    # features
    'arg_scope',  # req?
    'comments',  # opt?
    # endpoints
    'source',
    'target',
]


def read_game_as_dataframes(game_folder, sel_annotator=None, thorough=True):
    """Read an annotated game as dataframes.

    Parameters
    ----------
    game_folder : path
        Path to the game folder.

    sel_annotator : str, optional
        Identifier of the annotator whose version we want. If `None`,
        the existing metal annotator will be used (BRONZE|SILVER|GOLD).

    thorough : boolean, defaults to True
        If True, check that annotations in 'units' and 'unannotated'
        that are expected to have a strict equivalent in 'dialogue'
        actually do.

    Returns
    -------
    dfs : tuple of DataFrame
        DataFrames for the annotated game.
    """
    if sel_annotator is None:
        sel_annotator = 'metal'

    df_turns = []  # turns
    df_segs = []  # segments: EDUs, EEUs
    df_dlgs = []  # dialogues
    df_schms = []  # schemas: CDUs
    df_schm_mbrs = []  # schema members
    df_rels = []  # relations
    df_acts = []  # dialogue acts
    df_res = []  # resources
    df_pref = []  # preferences

    print(game_folder)  # DEBUG
    game_upfolder, game_name = os.path.split(game_folder)
    game_corpus = StacReader(game_upfolder).slurp(doc_glob=game_name)
    for doc_key, doc_val in sorted(game_corpus.items()):
        doc = doc_key.doc
        subdoc = doc_key.subdoc
        stage = doc_key.stage
        annotator = doc_key.annotator
        # skip docs not from a selected annotator
        if ((sel_annotator == 'metal' and
             annotator not in ('BRONZE', 'SILVER', 'GOLD')) or
            (sel_annotator != 'metal' and
             annotator != sel_annotator)):
            continue
        # process annotations in doc
        # print(doc, subdoc, stage, annotator)  # verbose
        doc_text = doc_val.text()
        # print(doc_text)
        for anno in doc_val.units:
            # attributes common to all units
            unit_dict = {
                # identification
                'global_id': anno.identifier(),
                'doc': doc,
                'subdoc': subdoc,
                'stage': stage,
                'annotator': annotator,
                # type, span, text
                'type': anno.type,
                'span_beg': anno.span.char_start,
                'span_end': anno.span.char_end,
                'text': doc_val.text(span=anno.span),
                # metadata
                'creation_date': anno.metadata['creation-date'],
                'author': anno.metadata['author'],
                # optional?
                'last_modifier': anno.metadata.get('lastModifier', None),
                'last_modif_date': anno.metadata.get('lastModificationDate', None),
            }

            # fields specific to each type of unit
            if is_paragraph(anno):
                # paragraph: ignore? one per turn
                pass
            elif is_turn(anno):
                # turn
                # comments = anno.features['Comments']
                # if comments == 'Please write in remarks...':
                unit_dict.update({
                    # features
                    'timestamp': anno.features['Timestamp'],
                    'comments': anno.features['Comments'],
                    'developments': anno.features['Developments'],
                    'turn_id': anno.features['Identifier'],
                    'emitter': anno.features['Emitter'],
                    'resources': anno.features['Resources'],
                })
                if stage == 'discourse':
                    df_turns.append(unit_dict)
                elif thorough:
                    pass  # FIXME check existence (exact duplicate)
            elif is_edu(anno):
                # segment: EDU or EEU
                if stage == 'discourse':
                    if anno.features:
                        raise ValueError('Wow, a discourse segment has *features*')
                    df_segs.append(unit_dict)
                elif stage == 'units':
                    # each entry (should) correspond to an entry in df_segs
                    act_dict = {
                        'global_id': anno.identifier(),  # foreign key
                        'surface_act': anno.features['Surface_act'],
                        'addressee': anno.features['Addressee'],
                    }
                    assert (sorted(anno.features.keys()) ==
                            ['Addressee', 'Surface_act'])
                    df_acts.append(act_dict)
                if thorough and stage in ('units', 'unannotated'):
                    # maybe metadata in 'units' has changed? eg. last
                    # modification date, last modifier
                    pass  # FIXME check existence (exact duplicate)
            elif is_dialogue(anno):
                expected_dlg_features = set(
                    ['Dice_rolling', 'Gets', 'Trades'])
                if set(anno.features.keys()).issubset(expected_dlg_features):
                    unit_dict.update({
                        # features
                        'gets': anno.features.get('Gets', None),
                        'trades': anno.features.get('Trades', None),
                        'dice_rolls': anno.features.get('Dice_rolling', None),
                    })
                else:
                    warn_msg = 'Dialogue {}: unexpected features {}'.format(
                        anno.identifier(),
                        ', '.join(x for x in sorted(anno.features.keys())
                                  if x not in set(expected_dlg_features)))
                    warnings.warn(warn_msg)

                if stage == 'discourse':
                    df_dlgs.append(unit_dict)
                elif thorough:
                    pass  # FIXME check existence (exact duplicate)
            elif is_resource(anno):
                unit_dict.update({
                    # features
                    'status': anno.features['Status'],
                    'kind': anno.features['Kind'],
                    'correctness': anno.features['Correctness'],
                    'quantity': anno.features['Quantity'],
                })
                assert (sorted(anno.features.keys()) ==
                        ['Correctness', 'Kind', 'Quantity', 'Status'])
                df_res.append(unit_dict)
            elif is_preference(anno):
                if anno.features:
                    print(anno.__dict__)
                    raise ValueError('Preference with features {}'.format(
                        anno.features))
                df_pref.append(unit_dict)
            else:
                print(anno.__dict__)
                raise ValueError('what unit is this?')
            # print('Unit', anno)

        for anno in doc_val.schemas:
            # in 'discourse': CDUs ;
            # in 'units': combinations of resources (OR, AND)
            schm_dict = {
                # identification
                'global_id': anno.identifier(),
                'doc': doc,
                'subdoc': subdoc,
                'stage': stage,
                'annotator': annotator,
                # type
                'type': anno.type,
                # metadata
                'creation_date': anno.metadata['creation-date'],
                'author': anno.metadata['author'],
                # optional? metadata
                'last_modifier': anno.metadata.get('lastModifier', None),
                'last_modif_date': anno.metadata.get('lastModificationDate', None),
            }
            # assumption: no feature
            if anno.features:
                if stage == 'units':
                    if anno.features.keys() == ['Operator']:
                        schm_dict.update({
                            'operator': anno.features['Operator'],
                        })
                    else:
                        print(anno.origin)
                        print(anno.__dict__)
                        print(anno.features)
                        raise ValueError('{}: schema with *features*'.format(
                            stage))
                elif stage == 'discourse':
                    # tolerate 'default': 'default' for the moment, but
                    # should probably cleaned out
                    if anno.features.keys() == ['default']:
                        schm_dict.update({
                            'default': anno.features['default'],
                        })
                    else:
                        print(anno.origin)
                        print(anno.__dict__)
                        print(anno.features)
                        raise ValueError('{}: schema with *features*'.format(
                            stage))
            df_schms.append(schm_dict)
            # associate to this schema each of its members ; assumptions:
            # - members should be units or schemas (no relation)
            if anno.relations:
                raise ValueError('Wow, a schema with *relation members*')
            for member in anno.members:
                member_dict = {
                    'member_id': member.identifier(),
                    'schema_id': anno.identifier(),
                }
                df_schm_mbrs.append(member_dict)
            # TODO post-verification: check that all members do exist
            # (should be useless as stac-check should catch it)
        for anno in doc_val.relations:
            # attributes common to all(?) types of annotations
            rel_dict = {
                # identification
                'global_id': anno.identifier(),
                'doc': doc,
                'subdoc': subdoc,
                'stage': stage,
                'annotator': annotator,
                # type
                'type': anno.type,
                # metadata
                'last_modifier': anno.metadata['lastModifier'],
                'last_modif_date': anno.metadata['lastModificationDate'],
                'creation_date': anno.metadata['creation-date'],
                'author': anno.metadata['author'],
            }
            # attributes specific to relations
            if 'Argument_scope' not in anno.features:
                # required feature
                w_msg = '{}: relation {} has no Argument_scope'.format(
                    str(doc_key), anno.identifier()
                )
                warnings.warn(w_msg)
            rel_dict.update({
                # features
                'arg_scope': anno.features.get('Argument_scope', None), # req
                'comments': anno.features.get('Comments', None),  # opt
                # endpoints
                'source': anno.source.identifier(),
                'target': anno.target.identifier(),
            })
            df_rels.append(rel_dict)

    # create dataframes
    df_turns = pd.DataFrame(df_turns, columns=TURN_COLS)
    df_dlgs = pd.DataFrame(df_dlgs, columns=DLG_COLS)
    df_segs = pd.DataFrame(df_segs, columns=SEG_COLS)
    df_acts = pd.DataFrame(df_acts, columns=ACT_COLS)
    df_schms = pd.DataFrame(df_schms, columns=SCHM_COLS)
    df_schm_mbrs = pd.DataFrame(df_schm_mbrs, columns=SCHM_MBRS_COLS)
    df_rels = pd.DataFrame(df_rels, columns=REL_COLS)
    df_res = pd.DataFrame(df_res, columns=RES_COLS)
    df_pref = pd.DataFrame(df_pref, columns=PREF_COLS)

    # add columns computed from other dataframes
    # * for segments: retrieve the turn_id and the char positions of the
    # beg and end of the segment in the turn text
    def get_seg_turn_cols(seg):
        """Helper to retrieve turn info for a segment (EDU, EEU)."""
        doc = seg['doc']
        subdoc = seg['subdoc']
        seg_beg = seg['span_beg']
        seg_end = seg['span_end']
        cand_turns = df_turns[(df_turns['span_beg'] <= seg_beg) &
                              (seg_end <= df_turns['span_end']) &
                              (doc == df_turns['doc']) &
                              (subdoc == df_turns['subdoc'])]
        # NB: cand_turns should contain a unique turn
        # compute the beg and end (char) positions of the segment in the turn
        # so we can match between the situated and linguistic versions when
        # the segmentation has changed
        turn_text = cand_turns['text'].item()
        seg_text = seg['text']
        turn_span_beg = turn_text.find(seg_text)
        turn_span_end = turn_span_beg + len(seg_text)
        turn_dict = {
            'turn_id': cand_turns['turn_id'].item(),
            'turn_span_beg': turn_span_beg,
            'turn_span_end': turn_span_end,
        }
        return pd.Series(turn_dict)

    seg_turn_cols = df_segs.apply(get_seg_turn_cols, axis=1)
    df_segs = pd.concat([df_segs, seg_turn_cols], axis=1)

    return (df_turns, df_dlgs, df_segs, df_acts, df_schms, df_schm_mbrs,
            df_rels, df_res, df_pref)


def read_corpus_as_dataframes(version='situated', split='all',
                              sel_games=None, exc_games=None):
    """Read the entire corpus as dataframes.

    Parameters
    ----------
    version : one of {'ling', 'situated'}, defaults to 'situated'
        Version of the corpus we want to examine.

    split : one of {'all', 'train', 'test'}, defaults to 'all'
        Split of the corpus.

    sel_games : list of str, optional
        List of selected games. If `None`, all games for the selected
        version and split.

    exc_games : list of str, optional
        List of excluded games. If `None`, all games for the selected
        version and split. Applies after, hence overrides, `sel_games`.

    Returns
    -------
    dfs : tuple of DataFrame
        Dataframes for turns, segments, acts...
    """
    if version not in ('ling', 'situated'):
        raise ValueError("Version must be one of {'ling', 'situated'}")
    if version == 'situated':
        base_dir = BASE_SITU
        all_globs = SPLIT_GLOBS
    else:
        base_dir = BASE_SPECT
        all_globs = SPLIT_GLOBS_SPECT

    if split not in ('all', 'train', 'test'):
        raise ValueError("Split must be one of {'all', 'train', 'test'}")
    if split == 'all':
        sel_globs = list(chain.from_iterable(all_globs.values()))
    elif split == 'train':
        sel_globs = all_globs['TRAIN']
    else:
        sel_globs = all_globs['TEST']

    sel_globs = [os.path.join(STAC_DATA_DIR, base_dir, x) for x in sel_globs]
    game_folders = list(chain.from_iterable(glob(x) for x in sel_globs))
    # map games to their folders
    game_dict = {os.path.basename(x): x for x in game_folders}
    if sel_games is not None:
        game_dict = {k: v for k, v in game_dict.items()
                     if k in sel_games}
    if exc_games is not None:
        game_dict = {k: v for k, v in game_dict.items()
                     if k not in exc_games}
    # lists of dataframes
    # TODO dataframe of docs? or glozz documents = subdocs?
    # what fields should be included?
    turn_dfs = []
    dlg_dfs = []
    seg_dfs = []
    act_dfs = []
    schm_dfs = []
    schm_mbr_dfs = []
    rel_dfs = []
    res_dfs = []
    pref_dfs = []
    for game_name, game_folder in game_dict.items():
        game_dfs = read_game_as_dataframes(game_folder)
        turn_dfs.append(game_dfs[0])
        dlg_dfs.append(game_dfs[1])
        seg_dfs.append(game_dfs[2])
        act_dfs.append(game_dfs[3])
        schm_dfs.append(game_dfs[4])
        schm_mbr_dfs.append(game_dfs[5])
        rel_dfs.append(game_dfs[6])
        res_dfs.append(game_dfs[7])
        pref_dfs.append(game_dfs[8])
    # concatenate each list into a single dataframe
    turns = pd.concat(turn_dfs, ignore_index=True)
    dlgs = pd.concat(dlg_dfs, ignore_index=True)
    segs = pd.concat(seg_dfs, ignore_index=True)
    acts = pd.concat(act_dfs, ignore_index=True)
    schms = pd.concat(schm_dfs, ignore_index=True)
    schm_mbrs = pd.concat(schm_mbr_dfs, ignore_index=True)
    rels = pd.concat(rel_dfs, ignore_index=True)
    res = pd.concat(res_dfs, ignore_index=True)
    pref = pd.concat(pref_dfs, ignore_index=True)
    return turns, dlgs, segs, acts, schms, schm_mbrs, rels, res, pref


if __name__ == '__main__':
    # situated games that are still incomplete, so should be excluded
    not_ready = ['s2-league3-game5', 's2-league4-game2']
    sel_games = None  # ['pilot20', 'pilot21']
    # read the situated version
    turns_situ, dlgs_situ, segs_situ, acts_situ, schms_situ, schm_mbrs_situ, rels_situ, res_situ, pref_situ = read_corpus_as_dataframes(version='situated', split='all', sel_games=sel_games, exc_games=not_ready)
    print(segs_situ[:5])
    if False:
        print(dlgs_situ[:5])
        print(segs_situ[:5])
        print(acts_situ[:5])
        print(schms_situ[:5])
        print(schm_mbrs_situ[:5])
        print(rels_situ[:5])
        print(res_situ[:5])
        print(pref_situ[:5])

    # get the list of documents in the situated version, filter _spect to keep
    # them (only)
    games_situ = list(turns_situ['doc'].unique())

    # read the spect version
    turns_spect, dlgs_spect, segs_spect, acts_spect, schms_spect, schm_mbrs_spect, rels_spect, res_spect, pred_spect = read_corpus_as_dataframes(version='ling', split='all', sel_games=games_situ)
    print(segs_spect[:5])
    if False:
        print(dlgs_spect[:5])
        print(acts_spect[:5])
        print(schms_spect[:5])
        print(schm_mbrs_spect[:5])
        print(rels_spect[:5])
        print(res_spect[:5])
        print(pref_spect[:5])

    # compare Dialog Act annotations between the two versions ; on common
    # turns, they should be (almost) 100% identical
    seg_acts_spect = pd.merge(segs_spect, acts_spect, on=['global_id'],
                              how='inner')
    seg_acts_situ = pd.merge(segs_situ, acts_situ, on=['global_id'],
                             how='inner')
    common_edus = pd.merge(
        seg_acts_situ, seg_acts_spect,
        on=['doc', 'turn_id', 'turn_span_beg', 'turn_span_end'],
        how='inner'
    )
    print('Common EDUs:',
          common_edus.shape[0], '/', seg_acts_spect.shape[0])
    print('>>>>>>>>>>><<<<<<<<<<<')
    print(common_edus[:5])
    diff_acts = ((common_edus['surface_act_x'] != common_edus['surface_act_y']) &
                 (common_edus['addressee_x'] != common_edus['addressee_y']))
    changed_edu_acts = common_edus[diff_acts]
    if changed_edu_acts.shape[0] > 0:
        print('Changed EDU acts:',
              changed_edu_acts.shape[0], '/', seg_acts_spect.shape[0])
        print(changed_edu_acts[
            ['doc', 'turn_id', 'turn_span_beg', 'turn_span_end',
             'subdoc_x', 'global_id_x', 'text_x',
             'surface_act_x', 'addressee_x',
             'subdoc_y', 'global_id_y', 'text_y',
             'surface_act_y', 'addressee_y']
        ][:15])
    else:
        print('No changed EDU acts')
