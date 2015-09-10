# pylint: disable=invalid-name
# pylint: disable=too-few-public-methods
"""
Feature extraction library functions for STAC corpora.
The feature extraction script (rel-info) is a lightweight frontend
to this library
"""

from __future__ import absolute_import, print_function
from collections import defaultdict, namedtuple
from functools import wraps
from itertools import chain
import collections
import copy
import itertools as itr
import os
import re
import sys

from nltk.corpus import verbnet as vnet
from soundex import Soundex

from educe.annotation import (Span)
from educe.external.parser import\
    SearchableTree,\
    ConstituencyTree
from educe.learning.keys import (MagicKey, Key, KeyGroup, MergedKeyGroup)
from educe.stac import postag, corenlp
from educe.stac.annotation import speaker, addressees, is_relation_instance
from educe.stac.context import (enclosed,
                                edus_in_span,
                                turns_in_span)
from educe.stac.corpus import (twin_key)
from educe.learning.csv import tune_for_csv
from educe.learning.util import tuple_feature, underscore
import educe.corpus
import educe.glozz
import educe.learning.csv as educe_csv
import educe.stac
import educe.stac.lexicon.pdtb_markers as pdtb_markers
import educe.stac.graph as stac_gr
import educe.util

from ..annotation import turn_id
from ..lexicon.wordclass import Lexicon
from ..fusion import (Dialogue, ROOT, FakeRootEDU,
                      fuse_edus)


class CorpusConsistencyException(Exception):
    """
    Exceptions which arise if one of our expecations about the
    corpus data is violated, in short, weird things we don't
    know how to handle. We should avoid using this for
    things which are definitely bugs in the code, and not just
    weird things in the corpus we didn't know how to handle.
    """
    def __init__(self, msg):
        super(CorpusConsistencyException, self).__init__(msg)


# ---------------------------------------------------------------------
# lexicon configuration
# ---------------------------------------------------------------------


class LexWrapper(object):
    """
    Configuration options for a given lexicon: where to find it,
    what to call it, what sorts of results to return
    """

    def __init__(self, key, filename, classes):
        """
        Note: classes=True means we want the (sub)-class of the lexical
        item found, and not just a general boolean
        """
        self.key = key
        self.filename = filename
        self.classes = classes
        self.lexicon = None


    def read(self, lexdir):
        """
        Read and store the lexicon as a mapping from words to their
        classes
        """
        path = os.path.join(lexdir, self.filename)
        self.lexicon = Lexicon.read_file(path)


LEXICONS = [LexWrapper('domain', 'stac_domain.txt', True),
            LexWrapper('robber', 'stac_domain2.txt', False),
            LexWrapper('trade', 'trade.txt', True),
            LexWrapper('dialog', 'dialog.txt', False),
            LexWrapper('opinion', 'opinion.txt', False),
            LexWrapper('modifier', 'modifiers.txt', False),
            # hand-extracted from trade prediction code, could
            # perhaps be merged with one of the other lexicons
            # fr.irit.stac.features.CalculsTraitsTache3
            LexWrapper('pronoun', 'pronouns.txt', True),
            LexWrapper('ref', 'stac_referential.txt', False)]

PDTB_MARKERS_BASENAME = 'pdtb_markers.txt'

VerbNetEntry = namedtuple("VerbNetEntry", "classname lemmas")

VERBNET_CLASSES = ['steal-10.5',
                   'get-13.5.1',
                   'give-13.1-1',
                   'want-32.1-1-1',
                   'want-32.1',
                   'exchange-13.6-1']

INQUIRER_BASENAME = 'inqtabs.txt'

INQUIRER_CLASSES = ['Positiv',
                    'Negativ',
                    'Pstv',
                    'Ngtv',
                    'NegAff',
                    'PosAff',
                    'If',
                    'TrnGain',  # maybe gain/loss words related
                    'TrnLoss',  # ...transactions
                    'TrnLw',
                    'Food',    # related to Catan resources?
                    'Tool',    # related to Catan resources?
                    'Region',  # related to Catan game?
                    'Route']   # related to Catan game

# ---------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------

def strip_cdus(corpus):
    """
    For all documents in a corpus, remove any CDUs and relink the
    document accordingly.  This mutates the corpus
    """
    for key in corpus:
        # replace all CDUs in links with their recursive heads
        graph = stac_gr.Graph.from_doc(corpus, key)
        graph.strip_cdus(sloppy=True)

# ---------------------------------------------------------------------
# relation queries
# ---------------------------------------------------------------------


def emoticons(tokens):
    "Given some tokens, return just those which are emoticons"
    return frozenset(token for token in tokens if token.tag == 'E')


def is_just_emoticon(tokens):
    "Return true if a sequence of tokens consists of a single emoticon"
    if not isinstance(tokens, collections.Sequence):
        raise TypeError("tokens must form a sequence")
    return bool(emoticons(tokens)) and len(tokens) == 1


def player_addresees(edu):
    """
    The set of people spoken to during an edu annotation.
    This excludes known non-players, like 'All', or '?', or 'Please choose...',
    """
    addr1 = addressees(edu) or frozenset()
    return frozenset(x for x in addr1 if x not in ['All', '?'])


def position_of_speaker_first_turn(edu):
    """
    Given an EDU context, determine the position of the first turn by that
    EDU's speaker relative to other turns in that dialogue.
    """
    edu_speaker = edu.speaker()
    # we can assume these are sorted
    for i, turn in enumerate(edu.dialogue_turns):
        if speaker(turn) == edu_speaker:
            return i
    oops = "Implementation error? No turns found which match speaker's turn"
    raise CorpusConsistencyException(oops)


def players_for_doc(corpus, kdoc):
    """
    Return the set of speakers/addressees associated with a document.

    In STAC, documents are semi-arbitrarily cut into sub-documents for
    technical and possibly ergonomic reasons, ie. meaningless as far as we are
    concerned.  So to find all speakers, we would have to search all the
    subdocuments of a single document. ::

        (Corpus, String) -> Set String
    """
    speakers = set()
    docs = [corpus[k] for k in corpus if k.doc == kdoc]
    for doc in docs:
        for anno in doc.units:
            if educe.stac.is_turn(anno):
                turn_speaker = speaker(anno)
                if turn_speaker:
                    speakers.add(turn_speaker)
            elif educe.stac.is_edu(anno):
                speakers.update(player_addresees(anno))
    return frozenset(speakers)


def clean_chat_word(token):
    """
    Given a word and its postag (educe PosTag representation)
    return a somewhat tidied up version of the word.

    * Sequences of the same letter greater than length 3 are
      shortened to just length three
    * Letter is lower cased
    """
    if token.tag == 'E':
        return token.word
    else:
        word = token.word.lower()
        # collapse 3 or more of the same char into 3
        return re.sub(r'(.)\1{2,}', r'\1\1\1', word)


def has_one_of_words(sought, tokens, norm=lambda x: x.lower()):
    """
    Given a set of words, a collection tokens, return True if the
    tokens contain words match one of the desired words, modulo
    some minor normalisations like lowercasing.
    """
    norm_sought = frozenset(norm(word) for word in sought)
    norm_tokens = frozenset(norm(tok.word) for tok in tokens)
    return bool(norm_sought & norm_tokens)


def has_pdtb_markers(markers, tokens):
    """
    Given a sequence of tagged tokens, return True
    if any of the given PDTB markers appears within the tokens
    """
    if not isinstance(tokens, collections.Sequence):
        raise TypeError("tokens must form a sequence")
    words = [t.word for t in tokens]
    return pdtb_markers.Marker.any_appears_in(markers, words)


def lexical_markers(lclass, tokens):
    """
    Given a dictionary (words to categories) and a text span, return all the
    categories of words that appear in that set.

    Note that for now we are doing our own white-space based tokenisation,
    but it could make sense to use a different source of tokens instead
    """
    sought = lclass.just_words()
    present = frozenset(t.word.lower() for t in tokens)
    return frozenset(lclass.word_to_subclass[x] for x in sought & present)


def real_dialogue_act(edu):
    """
    Given an EDU in the 'discourse' stage of the corpus, return its
    dialogue act from the 'units' stage
    """
    acts = educe.stac.dialogue_act(edu)
    if len(acts) < 1:
        oops = 'Was expecting at least one dialogue act for %s' % edu
        raise CorpusConsistencyException(oops)
    else:
        if len(acts) > 1:
            print("More than one dialogue act for %s: %s" % (edu, acts),
                  file=sys.stderr)
        return list(acts)[0]


def enclosed_lemmas(span, parses):
    """
    Given a span and a list of parses, return any lemmas that
    are within that span
    """
    return [x.features["lemma"] for x in enclosed(span, parses.tokens)]


def subject_lemmas(span, trees):
    """
    Given a span and a list of dependency trees, return any lemmas
    which are marked as being some subject in that span
    """
    def prunable(tree):
        "is outside the search span, so stop going down"
        return not span.overlaps(tree.span)

    def good(tree):
        "is within the search span"
        return tree.link == "nsubj" and\
            span.encloses(tree.label().text_span())

    subtrees = map_topdown(good, prunable, trees)
    return [tree.label().features["lemma"] for tree in subtrees]


def map_topdown(good, prunable, trees):
    """
    Do topdown search on all these trees, concatenate results.
    """
    return list(chain.from_iterable(
        tree.topdown(good, prunable)
        for tree in trees if isinstance(tree, SearchableTree)))


def enclosed_trees(span, trees):
    """
    Return the biggest (sub)trees in xs that are enclosed in the span
    """
    def prunable(tree):
        "is outside the search span, so stop going down"
        return not span.overlaps(tree.span)

    def good(tree):
        "is within the search span"
        return span.encloses(tree.span)

    return map_topdown(good, prunable, trees)

# ---------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------

# The comments on these named tuples can be docstrings in Python3,
# or we can wrap the class, but eh...

# feature extraction environment
DocEnv = namedtuple("DocEnv", "inputs current sf_cache")

# Global resources and settings used to extract feature vectors
FeatureInput = namedtuple('FeatureInput',
                          ['corpus', 'postags', 'parses',
                           'lexicons', 'pdtb_lex',
                           'verbnet_entries',
                           'inquirer_lex'])

# A document and relevant contextual information
DocumentPlus = namedtuple('DocumentPlus',
                          ['key',
                           'doc',
                           'unitdoc',  # equiv doc from units
                           'players',
                           'parses'])

# ---------------------------------------------------------------------
# feature decorators
# ---------------------------------------------------------------------


def type_text(wrapped):
    """
    Given a feature that emits text, clean its output up so to work
    with a wide variety of csv parsers ::

        (a -> String) ->
        (a -> String)
    """
    @wraps(wrapped)
    def inner(*args, **kwargs):
        "call the wrapped function"
        return tune_for_csv(wrapped(*args, **kwargs))
    return inner


def edu_text_feature(wrapped):
    """
    Lift a text based feature into a standard single EDU one ::

        (String -> a) ->
        ((Current, Edu) -> a)
    """
    @wraps(wrapped)
    def inner(current, edu):
        "call the wrapped fuction"
        txt = current.doc.text(edu.text_span())
        return wrapped(txt)
    return inner


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def clean_dialogue_act(act):
    """
    Knock out temporary markers used during corpus annotation
    """
    # pylint: disable=fixme
    pref = "FIXME:"
    # pylint: enable=fixme
    act2 = act[len(pref):] if act.startswith(pref) else act
    return "Other" if act2 in ["Strategic_comment",
                               "Preference"] else act2


# ---------------------------------------------------------------------
# single EDU non-lexical features
# ---------------------------------------------------------------------


def feat_id(_, edu):
    "some sort of unique identifier for the EDU"
    return edu.identifier()


def feat_start(_, edu):
    "text span start"
    return edu.text_span().char_start


def feat_end(_, edu):
    "text span end"
    return edu.text_span().char_end


def num_tokens(_, edu):
    "length of this EDU in tokens"
    return len(edu.tokens)


@type_text
def word_first(_, edu):
    "the first word in this EDU"
    return clean_chat_word(edu.tokens[0]) if edu.tokens else None


@type_text
def word_last(_, edu):
    "the last word in this EDU"
    return clean_chat_word(edu.tokens[-1]) if edu.tokens else None


def has_player_name_exact(current, edu):
    "if the EDU text has a player name in it"
    tokens = edu.tokens
    return has_one_of_words(current.players, tokens)


def has_player_name_fuzzy(current, edu):
    "if the EDU has a word that sounds like a player name"
    tokens = edu.tokens
    soundex = lambda w: Soundex().soundex(w)
    return has_one_of_words(current.players, tokens, norm=soundex)


def feat_has_emoticons(_, edu):
    "if the EDU has emoticon-tagged tokens"
    return bool(emoticons(edu.tokens))


def feat_is_emoticon_only(_, edu):
    "if the EDU consists solely of an emoticon"
    return is_just_emoticon(edu.tokens)


@edu_text_feature
def has_correction_star(text):
    "if the EDU begins with a '*' but does not contain others"
    return text[0] == "*" and "*" not in text[1:]


@edu_text_feature
def ends_with_bang(text):
    "if the EDU text ends with '!'"
    return text[-1] == '!'


@edu_text_feature
def ends_with_qmark(text):
    "if the EDU text ends with '?'"
    return text[-1] == '?'


@type_text
def lemma_subject(current, edu):
    "the lemma corresponding to the subject of this EDU"

    subjects = subject_lemmas(edu.text_span(),
                              current.parses.deptrees)
    return subjects[0] if subjects else None


def is_nplike(anno):
    "is some sort of NP annotation from a parser"
    return isinstance(anno, ConstituencyTree)\
        and anno.label() in ['NP', 'WHNP', 'NNP', 'NNPS']


def has_FOR_np(current, edu):
    "if the EDU has the pattern IN(for).. NP"

    def is_prep_for(anno):
        "is a node representing for as the prep in a PP"
        return isinstance(anno, ConstituencyTree)\
            and anno.label() == 'IN'\
            and len(anno.children) == 1\
            and anno.children[0].features["lemma"] == "for"

    def is_for_pp_with_np(anno):
        "is a for PP node (see above) with some NP-like descendant"
        return any(is_prep_for(child) for child in anno.children)\
            and anno.topdown(is_nplike, None)

    trees = enclosed_trees(edu.text_span(),
                           current.parses.trees)
    return bool(map_topdown(is_for_pp_with_np, None, trees))


QUESTION_WORDS = ["what",
                  "which",
                  "where",
                  "when",
                  "who",
                  "how",
                  "why",
                  "whose"]


def is_question(current, edu):
    "if the EDU is (or contains) a question"

    def is_sqlike(anno):
        "is some sort of question"
        return isinstance(anno, ConstituencyTree)\
            and anno.label() in ['SBARQ', 'SQ']

    doc = current.doc
    span = edu.text_span()
    has_qmark = "?" in doc.text(span)[-1]

    tokens = edu.tokens
    starts_w_qword = False
    if tokens:
        starts_w_qword = tokens[0].word.lower() in QUESTION_WORDS

    parses = current.parses
    trees = enclosed_trees(span, parses.trees)
    with_q_tag = map_topdown(is_sqlike, None, trees)
    has_q_tag = bool(with_q_tag)
    return has_qmark or starts_w_qword or has_q_tag


def edu_position_in_turn(_, edu):
    "relative position of the EDU in the turn"
    return 1 + edu.turn_edus.index(edu)


def position_in_dialogue(_, edu):
    "relative position of the turn in the dialogue"
    return 1 + edu.dialogue_turns.index(edu.turn)


def position_in_game(_, edu):
    "relative position of the turn in the game"
    return 1 + edu.doc_turns.index(edu.turn)


def turn_follows_gap(_, edu):
    "if the EDU turn number is > 1 + previous turn"
    tid = turn_id(edu.turn)
    dialogue_tids = [turn_id(x) for x in edu.dialogue_turns]
    foll_gap = not(tid and
                   tid - 1 in dialogue_tids and
                   tid != min(dialogue_tids))
    return foll_gap


def speaker_started_the_dialogue(_, edu):
    "if the speaker for this EDU is the same as that of the\
 first turn in the dialogue"
    return speaker(edu.dialogue_turns[0]) == speaker(edu.turn)


def speaker_already_spoken_in_dialogue(_, edu):
    "if the speaker for this EDU is the same as that of a\
 previous turn in the dialogue"
    return position_of_speaker_first_turn(edu) <\
        edu.dialogue_turns.index(edu.turn)


def speakers_first_turn_in_dialogue(_, edu):
    "position in the dialogue of the turn in which the\
 speaker for this EDU first spoke"
    return 1 + position_of_speaker_first_turn(edu)

# ---------------------------------------------------------------------
# pair features
# ---------------------------------------------------------------------


#pylint: disable=unused-argument
def feat_annotator(current, edu1, edu2):
    "annotator for the subdoc"
    anno = current.doc.origin.annotator
    return "none" if anno is None or anno is "" else anno
#pylint: enable=unused-argument


@tuple_feature(underscore)  # decorator does the pairing boilerplate
def is_question_pairs(_, cache, edu):
    "boolean tuple: if each EDU is a question"
    return cache[edu].get("is_question", False)


@tuple_feature(underscore)
def dialogue_act_pairs(current, _, edu):
    "tuple of dialogue acts for both EDUs"
    return clean_dialogue_act(real_dialogue_act(edu))


EduGap = namedtuple("EduGap", "sf_cache inner_edus turns_between")


#pylint: disable=unused-argument
def num_edus_between(_current, gap, _edu1, _edu2):
    "number of intervening EDUs (0 if adjacent)"
    return len(gap.inner_edus)


def num_speakers_between(_current, gap, _edu1, _edu2):
    "number of distinct speakers in intervening EDUs"
    return len(frozenset(speaker(t) for t in gap.turns_between))


def num_nonling_tstars_between(_current, gap, _edu1, _edu2):
    "number of non-linguistic turn-stars between EDUs"
    tid1 = (turn_id(_edu1.turn) if _edu1 != FakeRootEDU else
            min(turn_id(x) for x in _edu2.dialogue_turns) - 1)
    tid2 = turn_id(_edu2.turn)
    tids_span = [tid1] + [turn_id(t) for t in gap.turns_between] + [tid2]
    nb_nonling_tstars = 0
    for tid_i, tid_j in zip(tids_span[:-1], tids_span[1:]):
        if tid_j - tid_i > 1:
            nb_nonling_tstars += 1

    return nb_nonling_tstars


def has_inner_question(current, gap, _edu1, _edu2):
    "if there is an intervening EDU that is a question"
    return any(gap.sf_cache[x]["is_question"]
               for x in gap.inner_edus)
#pylint: enable=unused-argument


def same_speaker(current, _, edu1, edu2):
    "if both EDUs have the same speaker"
    return edu1.speaker() == edu2.speaker()


def same_turn(current, _, edu1, edu2):
    "if both EDUs are in the same turn"
    return edu1.turn == edu2.turn


# ---------------------------------------------------------------------
# single EDU lexical features
# ---------------------------------------------------------------------


class LexKeyGroup(KeyGroup):
    """
    The idea here is to provide a feature per lexical class in the
    lexicon entry
    """
    def __init__(self, lexicon):
        self.key = lexicon.key
        self.has_subclasses = lexicon.classes
        self.lexicon = lexicon.lexicon
        description = "%s (lexical features)" % self.key_prefix()
        super(LexKeyGroup, self).__init__(description,
                                          self.mk_fields())

    def mk_field(self, cname, subclass=None):
        """
        For a given lexical class, return the name of its feature in the
        CSV file
        """
        subclass_elems = [subclass] if subclass else []
        name = "_".join([self.key_prefix(), cname] + subclass_elems)
        helptxt = "boolean (subclass of %s)" % cname if subclass else "boolean"
        return Key.discrete(name, helptxt)

    def mk_fields(self):
        """
        CSV field names for each entry/class in the lexicon
        """
        if self.has_subclasses:
            headers = []
            for cname, lclass in self.lexicon.entries.items():
                headers.extend(self.mk_field(cname, subclass=x)
                               for x in lclass.just_subclasses())
            return headers
        else:
            return [self.mk_field(e) for e in self.lexicon.entries]

    def key_prefix(self):
        """
        Common CSV header name prefix to all columns based on this particular
        lexicon
        """
        return "lex_" + self.key

    def fill(self, current, edu, target=None):
        """
        See `SingleEduSubgroup`
        """
        vec = self if target is None else target
        tokens = edu.tokens
        for cname, lclass in self.lexicon.entries.items():
            markers = lexical_markers(lclass, tokens)
            if self.has_subclasses:
                for subclass in lclass.just_subclasses():
                    field = self.mk_field(cname, subclass)
                    vec[field.name] = subclass in markers
            else:
                field = self.mk_field(cname)
                vec[field.name] = bool(markers)


class PdtbLexKeyGroup(KeyGroup):
    """
    One feature per PDTB marker lexicon class
    """
    def __init__(self, lexicon):
        self.lexicon = lexicon
        description = "PDTB features"
        super(PdtbLexKeyGroup, self).__init__(description,
                                              self.mk_fields())

    def mk_field(self, rel):
        "From relation name to feature key"
        name = '_'.join([self.key_prefix(), rel])
        return Key.discrete(name, "pdtb " + rel)

    def mk_fields(self):
        "Feature name for each relation in the lexicon"
        return [self.mk_field(x) for x in self.lexicon]

    @classmethod
    def key_prefix(cls):
        "All feature keys in this lexicon should start with this string"
        return "pdtb"

    def fill(self, current, edu, target=None):
        "See `SingleEduSubgroup`"
        vec = self if target is None else target
        tokens = edu.tokens
        for rel in self.lexicon:
            field = self.mk_field(rel)
            has_marker = has_pdtb_markers(self.lexicon[rel], tokens)
            vec[field.name] = has_marker


class VerbNetLexKeyGroup(KeyGroup):
    """
    One feature per VerbNet lexicon class
    """
    def __init__(self, ventries):
        self.ventries = ventries
        description = "VerbNet features"
        super(VerbNetLexKeyGroup, self).__init__(description,
                                                 self.mk_fields())

    def mk_field(self, ventry):
        "From verb class to feature key"
        name = '_'.join([self.key_prefix(), ventry.classname])
        return Key.discrete(name, "VerbNet " + ventry.classname)

    def mk_fields(self):
        "Feature name for each relation in the lexicon"
        return [self.mk_field(x) for x in self.ventries]

    @classmethod
    def key_prefix(cls):
        "All feature keys in this lexicon should start with this string"
        return "verbnet"

    def fill(self, current, edu, target=None):
        "See `SingleEduSubgroup`"

        vec = self if target is None else target
        lemmas = frozenset(enclosed_lemmas(edu.text_span(), current.parses))
        for ventry in self.ventries:
            matching = lemmas.intersection(ventry.lemmas)
            field = self.mk_field(ventry)
            vec[field.name] = bool(matching)


class InquirerLexKeyGroup(KeyGroup):
    """
    One feature per Inquirer lexicon class
    """
    def __init__(self, lexicon):
        self.lexicon = lexicon
        description = "Inquirer features"
        super(InquirerLexKeyGroup, self).__init__(description,
                                                  self.mk_fields())

    def mk_field(self, entry):
        "From verb class to feature key"
        name = '_'.join([self.key_prefix(), entry])
        return Key.discrete(name, "Inquirer " + entry)

    def mk_fields(self):
        "Feature name for each relation in the lexicon"
        return [self.mk_field(x) for x in self.lexicon]

    @classmethod
    def key_prefix(cls):
        "All feature keys in this lexicon should start with this string"
        return "inq"

    def fill(self, current, edu, target=None):
        "See `SingleEduSubgroup`"

        vec = self if target is None else target
        tokens = frozenset(t.word.lower() for t in edu.tokens)
        for entry in self.lexicon:
            field = self.mk_field(entry)
            matching = tokens.intersection(self.lexicon[entry])
            vec[field.name] = bool(matching)


class MergedLexKeyGroup(MergedKeyGroup):
    """
    Single-EDU features based on lexical lookup.
    """
    def __init__(self, inputs):
        groups =\
            [LexKeyGroup(l) for l in inputs.lexicons] +\
            [PdtbLexKeyGroup(inputs.pdtb_lex),
             InquirerLexKeyGroup(inputs.inquirer_lex),
             VerbNetLexKeyGroup(inputs.verbnet_entries)]
        description = "lexical features"
        super(MergedLexKeyGroup, self).__init__(description, groups)

    def fill(self, current, edu, target=None):
        "See `SingleEduSubgroup`"
        for group in self.groups:
            group.fill(current, edu, target)


# ---------------------------------------------------------------------
# single EDU non-lexical feature groups
# ---------------------------------------------------------------------


class SingleEduSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged SingleEduKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(SingleEduSubgroup, self).__init__(description, keys)

    def fill(self, current, edu, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)

        This defaults to _magic_fill if you don't implement it.
        """
        self._magic_fill(current, edu, target)

    def _magic_fill(self, current, edu, target=None):
        """
        Possible fill implementation that works on the basis of
        features defined wholly as magic keys
        """
        vec = self if target is None else target
        for key in self.keys:
            vec[key.name] = key.function(current, edu)


class SingleEduSubgroup_Token(SingleEduSubgroup):
    """
    word/token-based features
    """
    def __init__(self):
        desc = self.__doc__.strip()
        keys =\
            [MagicKey.continuous_fn(num_tokens),
             MagicKey.discrete_fn(word_first),
             MagicKey.discrete_fn(word_last),
             MagicKey.discrete_fn(has_player_name_exact)]
        if not sys.version > '3':
            keys.append(MagicKey.discrete_fn(has_player_name_fuzzy))
        keys2 =\
            [MagicKey.discrete_fn(feat_has_emoticons),
             MagicKey.discrete_fn(feat_is_emoticon_only)]
        keys.extend(keys2)
        super(SingleEduSubgroup_Token, self).__init__(desc, keys)


class SingleEduSubgroup_Punct(SingleEduSubgroup):
    "punctuation features"

    def __init__(self):
        desc = self.__doc__.strip()
        keys =\
            [MagicKey.discrete_fn(has_correction_star),
             MagicKey.discrete_fn(ends_with_bang),
             MagicKey.discrete_fn(ends_with_qmark)]
        super(SingleEduSubgroup_Punct, self).__init__(desc, keys)


class SingleEduSubgroup_Chat(SingleEduSubgroup):
    """
    Single-EDU features based on the EDU's relationship with the
    the chat structure (eg turns, dialogues).
    """

    def __init__(self):
        desc = "chat history features"
        keys =\
            [MagicKey.discrete_fn(speaker_started_the_dialogue),
             MagicKey.discrete_fn(speaker_already_spoken_in_dialogue),
             MagicKey.continuous_fn(speakers_first_turn_in_dialogue),
             MagicKey.discrete_fn(turn_follows_gap),
             MagicKey.continuous_fn(position_in_dialogue),
             MagicKey.continuous_fn(position_in_game),
             MagicKey.continuous_fn(edu_position_in_turn)]
        super(SingleEduSubgroup_Chat, self).__init__(desc, keys)


class SingleEduSubgroup_Parser(SingleEduSubgroup):
    """
    Single-EDU features that come out of a syntactic parser.
    """

    def __init__(self):
        desc = "parser features"
        keys =\
            [MagicKey.discrete_fn(lemma_subject),
             MagicKey.discrete_fn(has_FOR_np),
             MagicKey.discrete_fn(is_question)]
        super(SingleEduSubgroup_Parser, self).__init__(desc, keys)


class SingleEduKeys(MergedKeyGroup):
    """
    Features for a single EDU
    """
    def __init__(self, inputs):
        groups = [SingleEduSubgroup_Token(),
                  SingleEduSubgroup_Chat(),
                  SingleEduSubgroup_Punct(),
                  SingleEduSubgroup_Parser(),
                  MergedLexKeyGroup(inputs)]
        super(SingleEduKeys, self).__init__("single EDU features",
                                            groups)

    def fill(self, current, edu, target=None):
        """
        See `SingleEduSubgroup.fill`
        """
        vec = self if target is None else target
        for group in self.groups:
            group.fill(current, edu, vec)

# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

class PairSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged PairKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(PairSubgroup, self).__init__(description, keys)

    def fill(self, current, edu1, edu2, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)
        """
        raise NotImplementedError("fill should be implemented by a subclass")


class PairSubgroup_Tuple(PairSubgroup):
    "artificial tuple features"

    def __init__(self, inputs, sf_cache):
        self.corpus = inputs.corpus
        self.sf_cache = sf_cache
        desc = self.__doc__.strip()
        keys =\
            [MagicKey.discrete_fn(is_question_pairs),
             MagicKey.discrete_fn(dialogue_act_pairs)]
        super(PairSubgroup_Tuple, self).__init__(desc, keys)

    def fill(self, current, edu1, edu2, target=None):
        vec = self if target is None else target
        for key in self.keys:
            vec[key.name] = key.function(current, self.sf_cache, edu1, edu2)


class PairSubgroup_Gap(PairSubgroup):
    """
    Features related to the combined surrounding context of the
    two EDUs
    """

    def __init__(self, sf_cache):
        self.sf_cache = sf_cache
        desc = "the gap between EDUs"
        keys =\
            [MagicKey.continuous_fn(num_edus_between),
             MagicKey.continuous_fn(num_speakers_between),
             MagicKey.continuous_fn(num_nonling_tstars_between),
             MagicKey.discrete_fn(same_speaker),
             MagicKey.discrete_fn(same_turn),
             MagicKey.discrete_fn(has_inner_question)]
        super(PairSubgroup_Gap, self).__init__(desc, keys)

    def fill(self, current, edu1, edu2, target=None):
        vec = self if target is None else target
        doc = current.doc
        big_span = edu1.text_span().merge(edu2.text_span())

        # spans for the turns that come between the two edus
        turns_between_span = Span(edu1.turn.text_span().char_end,
                                  edu2.turn.text_span().char_start)
        turns_between = turns_in_span(doc, turns_between_span)

        inner_edus = edus_in_span(doc, big_span)
        if edu1.identifier() != ROOT: # not present anyway
            inner_edus.remove(edu1)
        if edu2.identifier() != ROOT:
            inner_edus.remove(edu2)

        gap = EduGap(inner_edus=inner_edus,
                     turns_between=turns_between,
                     sf_cache=self.sf_cache)

        for key in self.keys:
            vec[key.name] = key.function(current, gap, edu1, edu2)


class PairKeys(MergedKeyGroup):
    """
    Features for pairs of EDUs
    """
    def __init__(self, inputs, sf_cache=None):
        self.sf_cache = sf_cache
        groups = [PairSubgroup_Gap(sf_cache),
                  PairSubgroup_Tuple(inputs, sf_cache)]
        if sf_cache is None:
            self.edu1 = SingleEduKeys(inputs)
            self.edu2 = SingleEduKeys(inputs)
        else:
            self.edu1 = None  # will be filled out later
            self.edu2 = None  # from the feature cache

        super(PairKeys, self).__init__("pair features",
                                       groups)

    def one_hot_values_gen(self, suffix=''):
        for pair in super(PairKeys, self).one_hot_values_gen():
            yield pair
        for pair in self.edu1.one_hot_values_gen(suffix='_DU1'):
            yield pair
        for pair in self.edu2.one_hot_values_gen(suffix='_DU2'):
            yield pair

    def fill(self, current, edu1, edu2, target=None):
        "See `PairSubgroup`"
        vec = self if target is None else target
        vec.edu1 = self.sf_cache[edu1]
        vec.edu2 = self.sf_cache[edu2]
        for group in self.groups:
            group.fill(current, edu1, edu2, vec)

# ---------------------------------------------------------------------
# (single) feature cache
# ---------------------------------------------------------------------


class FeatureCache(dict):
    """
    Cache for single edu features.
    Retrieving an item from the cache lazily computes/memoises
    the single EDU features for it.
    """
    def __init__(self, inputs, current):
        self.inputs = inputs
        self.current = current
        super(FeatureCache, self).__init__()

    def __getitem__(self, edu):
        if edu.identifier() == ROOT:
            return KeyGroup('fake root group', [])
        elif edu in self:
            return super(FeatureCache, self).__getitem__(edu)
        else:
            vec = SingleEduKeys(self.inputs)
            vec.fill(self.current, edu)
            self[edu] = vec
            return vec

    def expire(self, edu):
        """
        Remove an edu from the cache if it's in there
        """
        if edu in self:
            del self[edu]

# ---------------------------------------------------------------------
# extraction generators
# ---------------------------------------------------------------------


def _get_unit_key(inputs, key):
    """
    Given the key for what is presumably a discourse level or
    unannotated document, return the key for for its unit-level
    equivalent.
    """
    if key.annotator is None:
        twins = [k for k in inputs.corpus if
                 k.doc == key.doc and
                 k.subdoc == key.subdoc and
                 k.stage == 'units']
        return twins[0] if twins else None
    else:
        twin = copy.copy(key)
        twin.stage = 'units'
        return twin if twin in inputs.corpus else None


def mk_env(inputs, people, key):
    """
    Pre-process and bundle up a representation of the current document
    """
    doc = inputs.corpus[key]
    unit_key = _get_unit_key(inputs, key)
    current =\
        DocumentPlus(key=key,
                     doc=doc,
                     unitdoc=inputs.corpus[unit_key] if unit_key else None,
                     players=people[key.doc],
                     parses=inputs.parses[key] if inputs.parses else None)

    return DocEnv(inputs=inputs,
                  current=current,
                  sf_cache=FeatureCache(inputs, current))


def get_players(inputs):
    """
    Return a dictionary mapping each document to the set of
    players in that document
    """
    kdocs = frozenset(k.doc for k in inputs.corpus)
    return {x: players_for_doc(inputs.corpus, x)
            for x in kdocs}


def relation_dict(doc, quiet=False):
    """
    Return the relations instances from a document in the
    form of an id pair to label dictionary

    If there is more than one relation between a pair of
    EDUs we pick one of them arbitrarily and ignore the
    other
    """
    relations = {}
    for rel in doc.relations:
        if not is_relation_instance(rel):
            # might be the odd Anaphora link lying around
            continue
        pair = rel.source.identifier(), rel.target.identifier()
        if pair not in relations:
            relations[pair] = rel.type
        elif not quiet:
            print(('Ignoring {type1} relation instance ({edu1} -> {edu2}); '
                   'another of type {type2} already exists'
                   '').format(type1=rel.type,
                              edu1=pair[0],
                              edu2=pair[1],
                              type2=relations[pair]),
                  file=sys.stderr)
    # generate fake root links
    for anno in doc.units:
        if not educe.stac.is_edu(anno):
            continue
        is_target = False
        for rel in doc.relations:
            if rel.target == anno:
                is_target = True
                break
        if not is_target:
            key = ROOT, anno.identifier()
            relations[key] = ROOT
    return relations


def _extract_pair(env, edu1, edu2):
    """
    Extraction for a given pair of EDUs
    (directional, so would have to be called twice)
    """
    vec = PairKeys(env.inputs, sf_cache=env.sf_cache)
    vec.fill(env.current, edu1, edu2)
    return vec


def _id_pair(pair):
    "pair of ids for pair of edus"
    edu1, edu2 = pair
    return edu1.identifier(), edu2.identifier()


def _mk_high_level_dialogues(current):
    """
    Returns
    -------
    iterator of `educe.stac.fusion.Dialogue`
    """
    doc = current.doc
    # first pass: create the EDU objects
    edus = sorted([x for x in doc.units if educe.stac.is_edu(x)],
                  key=lambda x: x.span)
    edus_in_dialogues = defaultdict(list)
    for edu in edus:
        edus_in_dialogues[edu.dialogue].append(edu)

    # finally, generat the high level dialogues
    relations = relation_dict(doc)
    dialogues = sorted(edus_in_dialogues, key=lambda x: x.span)
    for dia in dialogues:
        d_edus = edus_in_dialogues[dia]
        d_relations = {}
        for pair in itr.product([FakeRootEDU] + d_edus, d_edus):
            rel = relations.get(_id_pair(pair))
            if rel is not None:
                d_relations[pair] = rel
        yield Dialogue(dia, d_edus, d_relations)


def mk_envs(inputs, stage):
    """
    Generate an environment for each document in the corpus
    within the given stage.

    The environment pools together all the information we
    have on a single document
    """
    people = get_players(inputs)
    for key in inputs.corpus:
        if key.stage != stage:
            continue
        yield mk_env(inputs, people, key)


def mk_high_level_dialogues(inputs, stage):
    """
    Generate all relevant EDU pairs for a document
    (generator)
    """
    for env in mk_envs(inputs, stage):
        for dia in _mk_high_level_dialogues(env.current):
            yield dia


def extract_pair_features(inputs, stage):
    """
    Extraction for all relevant pairs in a document
    (generator)
    """
    for env in mk_envs(inputs, stage):
        for dia in _mk_high_level_dialogues(env.current):
            for edu1, edu2 in dia.edu_pairs():
                yield _extract_pair(env, edu1, edu2)

# ---------------------------------------------------------------------
# extraction generators (single edu)
# ---------------------------------------------------------------------


def extract_single_features(inputs, stage):
    """
    Return a dictionary for each EDU
    """
    for env in mk_envs(inputs, stage):
        doc = env.current.doc
        # skip any documents which are not yet annotated
        if env.current.unitdoc is None:
            continue
        edus = [unit for unit in doc.units if educe.stac.is_edu(unit)]
        for edu in edus:
            vec = SingleEduKeys(env.inputs)
            vec.fill(env.current, edu)
            yield vec

# ---------------------------------------------------------------------
# input readers
# ---------------------------------------------------------------------


def read_pdtb_lexicon(args):
    """
    Read and return the local PDTB discourse marker lexicon.
    """
    pdtb_lex_file = os.path.join(args.resources, PDTB_MARKERS_BASENAME)
    return pdtb_markers.read_lexicon(pdtb_lex_file)


def _read_inquirer_lexicon(args):
    """
    Read and return the local PDTB discourse marker lexicon.
    """
    inq_txt_file = os.path.join(args.resources, INQUIRER_BASENAME)
    with open(inq_txt_file) as cin:
        creader = educe_csv.SparseDictReader(cin, delimiter='\t')
        words = defaultdict(list)
        for row in creader:
            for k in row:
                word = row["Entry"].lower()
                word = re.sub(r'#.*$', r'', word)
                if k in INQUIRER_CLASSES:
                    words[k].append(word)
    return words


def mk_is_interesting(args, single):
    """
    Return a function that filters corpus keys to pick out the ones
    we specified on the command line

    We have two cases here: for pair extraction, we just want to
    grab the units and if possible the discourse stage. In live mode,
    there won't be a discourse stage, but that's fine because we can
    just fall back on units.

    For single extraction (dialogue acts), we'll also want to grab the
    units stage and fall back to unannotated when in live mode. This
    is made a bit trickier by the fact that unannotated does not have
    an annotator, so we have to accomodate that.

    Phew.

    It's a bit specific to feature extraction in that here we are
    trying

    :type single: bool
    """
    if single:
        # ignore annotator filter for unannotated documents
        args1 = copy.copy(args)
        args1.annotator = None
        is_interesting1 =\
            educe.util.mk_is_interesting(args1,
                                         preselected={'stage': ['unannotated']})
        # but pay attention to it for units
        args2 = args
        is_interesting2 =\
            educe.util.mk_is_interesting(args2,
                                         preselected={'stage': ['units']})
        return lambda x: is_interesting1(x) or is_interesting2(x)
    else:
        preselected = {"stage": ["discourse", "units"]}
        return educe.util.mk_is_interesting(args, preselected=preselected)


def _fuse_corpus(corpus, postags):
    "Merge any dialogue/unit level documents together"
    to_delete = []
    for key in corpus:
        if key.stage == 'unannotated':
            # slightly abusive use of fuse_edus to just get the effect of
            # having EDUs that behave like contexts
            #
            # context: feature extraction for live mode dialogue acts
            # extraction, so by definition we don't have a units stage
            corpus[key] = fuse_edus(corpus[key], corpus[key], postags[key])
        elif key.stage == 'units':
            # similar Context-only abuse of fuse-edus (here, we have a units
            # stage but no dialogue to make use of)
            #
            # context: feature extraction for
            # - live mode discourse parsing (by definition we don't have a
            #   discourse stage yet, but we might have a units stage
            #   inferred earlier in the parsing pipeline)
            # - dialogue act annotation from corpus
            corpus[key] = fuse_edus(corpus[key], corpus[key], postags[key])
        elif key.stage == 'discourse':
            ukey = twin_key(key, 'units')
            corpus[key] = fuse_edus(corpus[key], corpus[ukey], postags[key])
            to_delete.append(ukey)
    for key in to_delete:
        del corpus[key]


def read_corpus_inputs(args):
    """
    Read and filter the part of the corpus we want features for
    """
    reader = educe.stac.Reader(args.corpus)
    anno_files = reader.filter(reader.files(),
                               mk_is_interesting(args, args.single))
    corpus = reader.slurp(anno_files, verbose=True)

    if not args.ignore_cdus:
        strip_cdus(corpus)
    postags = postag.read_tags(corpus, args.corpus)
    parses = corenlp.read_results(corpus, args.corpus)
    _fuse_corpus(corpus, postags)

    for lex in LEXICONS:
        lex.read(args.resources)
    pdtb_lex = read_pdtb_lexicon(args)
    inq_lex = {} #_read_inquirer_lexicon(args)

    verbnet_entries = [VerbNetEntry(x, frozenset(vnet.lemmas(x)))
                       for x in VERBNET_CLASSES]

    return FeatureInput(corpus=corpus,
                        postags=postags,
                        parses=parses,
                        lexicons=LEXICONS,
                        pdtb_lex=pdtb_lex,
                        verbnet_entries=verbnet_entries,
                        inquirer_lex=inq_lex)
