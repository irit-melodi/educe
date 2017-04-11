"""This module provides a growing library of functions to quantitatively
examine the situated version of the STAC corpus, in itself and with
respect to the purely linguistic (spect) version.
"""

from __future__ import absolute_import, print_function

from glob import glob
from itertools import chain
import os

import pandas as pd

from educe.stac.annotation import (is_dialogue, is_edu, is_paragraph,
                                   is_resource, is_turn)
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
        print(doc, subdoc, stage, annotator, doc_val)
        doc_text = doc_val.text()
        print(doc_text)
        for anno in doc_val.units:
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
                'last_modifier': anno.metadata['lastModifier'],
                'last_modif_date': anno.metadata['lastModificationDate'],
                'creation_date': anno.metadata['creation-date'],
                'author': anno.metadata['author'],
            }
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
                if (sorted(anno.features.keys()) !=
                    ['Dice_rolling', 'Gets', 'Trades']):
                    print(anno.features)  # RESUME HERE !
                    raise ValueError('missing features in dialogue')
                unit_dict.update({
                    # features
                    'gets': anno.features['Gets'],
                    'trades': anno.features['Trades'],
                    'dice_rolls': anno.features['Dice_rolling'],
                })
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
            else:
                print(anno.__dict__)
                raise ValueError('what unit is this?')
            # print('Unit', anno)
        for anno in doc_val.schemas:  # only in 'discourse' ?
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
                raise ValueError('Wow, a schema with *features*')
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
            rel_dict.update({
                # features
                'arg_scope': anno.features['Argument_scope'],
                'comments': anno.features['Comments'],
                # endpoints
                'source': anno.source.identifier(),
                'target': anno.target.identifier(),
            })
            # hyp: features has only these 2 fields
            assert (sorted(anno.features.keys()) ==
                    ['Argument_scope', 'Comments'])
            df_rels.append(rel_dict)

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
    for game_name, game_folder in game_dict.items():
        game_dfs = read_game_as_dataframes(game_folder)
    print(game_dict)


if __name__ == '__main__':
    read_corpus_as_dataframes(version='situated', split='all')
