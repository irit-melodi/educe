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


def read_game_as_dataframes(game_folder, thorough=True):
    """Read an annotated game as dataframes.

    Parameters
    ----------
    game_folder : path
        Path to the game folder.

    thorough : boolean, defaults to True
        If True, check that annotations in 'units' and 'unannotated'
        that are expected to have a strict equivalent in 'dialogue'
        actually do.

    Returns
    -------
    dfs : tuple of DataFrame
        DataFrames for the annotated game.
    """
    df_turns = []  # turns
    df_segs = []  # segments: EDUs, EEUs
    df_dlgs = []  # dialogues
    df_schms = []  # schemas: CDUs
    df_schm_mbrs = []  # schema members
    df_rels = []  # relations
    df_acts = []  # dialogue acts

    print(game_folder)  # DEBUG
    game_upfolder, game_name = os.path.split(game_folder)
    game_corpus = StacReader(game_upfolder).slurp(doc_glob=game_name)
    for doc_key, doc_val in sorted(game_corpus.items()):
        doc = doc_key.doc
        subdoc = doc_key.subdoc
        stage = doc_key.stage
        annotator = doc_key.annotator
        # process annotations in doc
        print(doc, subdoc, stage, annotator)
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
            }
            # additional metadata, sometimes absent
            if (('lastModificationDate' in anno.metadata and
                 'lastModifier' in anno.metadata)):
                unit_dict.update({
                    'last_modifier': anno.metadata['lastModifier'],
                    'last_modif_date': anno.metadata['lastModificationDate'],
                })

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
                    act_dict = {
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
            elif is_preference(anno):
                if anno.features:
                    print(anno.__dict__)
                    raise ValueError('Preference with features {}'.format(
                        anno.features))
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
                'last_modifier': anno.metadata['lastModifier'],
                'last_modif_date': anno.metadata['lastModificationDate'],
                'creation_date': anno.metadata['creation-date'],
                'author': anno.metadata['author'],
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
    df_turns = pd.DataFrame(df_turns)
    df_dlgs = pd.DataFrame(df_dlgs)
    df_segs = pd.DataFrame(df_segs)
    df_acts = pd.DataFrame(df_acts)
    df_schms = pd.DataFrame(df_schms)
    df_schm_mbrs = pd.DataFrame(df_schm_mbrs)
    df_rels = pd.DataFrame(df_rels)

    return (df_turns, df_dlgs, df_segs, df_acts, df_schms, df_schm_mbrs,
            df_rels)


def read_corpus_as_dataframes(version='situated', split='all'):
    """Read the entire corpus as dataframes.

    Parameters
    ----------
    version : one of {'ling', 'situated'}, defaults to 'situated'
        Version of the corpus we want to examine.

    split : one of {'all', 'train', 'test'}, defaults to 'all'
        Split of the corpus.
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
    # lists of dataframes
    turn_dfs = []
    dlg_dfs = []
    seg_dfs = []
    act_dfs = []
    schm_dfs = []
    schm_mbr_dfs = []
    rel_dfs = []
    for game_name, game_folder in game_dict.items():
        game_dfs = read_game_as_dataframes(game_folder)
        turn_dfs.append(game_dfs[0])
        dlg_dfs.append(game_dfs[1])
        seg_dfs.append(game_dfs[2])
        act_dfs.append(game_dfs[3])
        schm_dfs.append(game_dfs[4])
        schm_mbr_dfs.append(game_dfs[5])
        rel_dfs.append(game_dfs[6])
    # concatenate each list into a single dataframe
    turns = pd.concat(turn_dfs)
    dlgs = pd.concat(dlg_dfs)
    segs = pd.concat(seg_dfs)
    acts = pd.concat(act_dfs)
    schms = pd.concat(schm_dfs)
    schm_mbrs = pd.concat(schm_mbr_dfs)
    rels = pd.concat(rel_dfs)
    return turns, dlgs, segs, acts, schms, schm_mbrs, rels


if __name__ == '__main__':
    turns, dlgs, segs, acts, schms, schm_mbrs, rels = read_corpus_as_dataframes(version='situated', split='all')
    print(turns[:5])
    print(dlgs[:5])
    print(segs[:5])
    print(acts[:5])
    print(schms[:5])
    print(schm_mbrs[:5])
    print(rels[:5])
