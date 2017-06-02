r"""Lexicon of discourse markers.

Cheap and cheerful phrasal lexicon format used in the STAC project.
Maps sequences of multiword expressions to relations they mark

     as            ; explanation explanation* background
     as a result   ; result result*
     for example   ; elaboration
     if:then       ; conditional
     on the one hand:on the other hand

One entry per line.  Sometimes you have split expressions, like
"on the one hand X, on the other hand Y" (we model this by saying
that we are working with sequences of expressions, rather than
single expressions).  Phrases can be associated with 0 to N
relations (interpreted as disjunction; if `\wedge` appears (LaTeX
for the "logical and" operator), it is ignored).
"""

from __future__ import print_function
import codecs
from collections import defaultdict
from os.path import join, dirname


PDTB_MARKERS_FILE = join(dirname(__file__), 'pdtb_markers.txt')


class Multiword(object):
    """
    A sequence of tokens representing a multiword expression.
    """
    def __init__(self, words):
        self.words = [w.lower() for w in words]

    def __str__(self):
        return " ".join(self.words)


# TODO: We need to implement comparison/hashing functions so that objects with
# same contents are treated as the same. I miss Haskell
class Marker(object):
    """
    A marker here is a sort of template consisting of multiword expressions
    and holes, eg. "on the one hand, XXX, on the other hand YYY". We
    represent this is as a sequence of Multiword
    """
    def __init__(self, exprs):
        self.exprs = exprs

    def __str__(self):
        return " ... ".join(str(e) for e in self.exprs)

    def appears_in(self, words, sep='#####'):
        """
        Given a sequence of words, return True if this marker appears in
        that sequence.

        We use a *very* liberal defintion here. In particular, if the marker
        has more than component (on the one hand X, on the other hand Y),
        we merely check that all components appear without caring what order
        they appear in.

        Note that this abuses the Python string matching functionality,
        and assumes that the separator substring never appears in the
        tokens
        """
        # add leading and trailing empty word '', which results in
        # leading and trailing separators ;
        # the objective is to avoid erroneous matchings on
        # subwords, for example we don't want the marker "as" to
        # match a substring of the string "has"
        sentence = sep.join([''] + words + ['']).lower()
        exprs = frozenset(sep.join([''] + e.words + [''])
                          for e in self.exprs)
        return all(sentence.find(e) >= 0 for e in exprs)

    @classmethod
    def any_appears_in(cls, markers, words, sep='#####'):
        """
        Return True if any of the given markers appears in the word
        sequence.

        See `appears_in` for details.
        """
        sentence = sep.join(words).lower()
        for m in markers:
            exprs = frozenset(sep.join(e.words) for e in m.exprs)
            if all(sentence.find(e) >= 0 for e in exprs):
                return True
        return False


def load_pdtb_markers_lexicon(filename):
    """Load the lexicon of discourse markers from the PDTB.

    Parameters
    ----------
    filename : str
        Path to the lexicon.

    Returns
    -------
    markers : dict(Marker, list(string))
        Discourse markers and the relations they signal
    """
    blacklist = frozenset(['\\wedge'])
    marker2rels = dict()  # result

    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = [fld.strip() for fld in line.split(';')]
            if len(fields) > 2:
                raise ValueError("Cannot parse PDTB marker entry: %s" % line)
            # first field: marker
            subexprs = [Multiword(se.strip().split())
                        for se in fields[0].split(':')]
            marker = Marker(subexprs)
            # second field (optional): possible signalled relations
            if len(fields) == 2:
                rels = frozenset(fields[1].split()) - blacklist
            else:
                rels = frozenset([])
            # store mapping
            marker2rels[marker] = rels
    return marker2rels


def read_lexicon(filename):
    """Load the lexicon of discourse markers from the PDTB, by relation.

    This calls `load_pdtb_markers_lexicon` but inverts the indexing to
    map each relation to its possible discourse markers.

    Note that, as an effect of this inversion, discourse markers whose
    set of relations is left empty in the lexicon (possibly because they
    are too ambiguous?) are absent from the inverted index.

    Parameters
    ----------
    filename : str
        Path to the lexicon.

    Returns
    -------
    relations : dict(string, frozenset(Marker))
        Relations and their signalling discourse markers.
    """
    rel2markers = defaultdict(list)
    # compute the inverse mapping; marker2rels -> rel2markers
    marker2rels = load_pdtb_markers_lexicon(filename)
    for marker, rels in marker2rels.items():
        for rel in rels:
            rel2markers[rel].append(marker)
    # store markers in a frozenset
    relations = {rel: frozenset(markers)
                 for rel, markers in rel2markers.items()}
    return relations
