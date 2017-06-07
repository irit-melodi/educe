"""
Low-level representation of corpus annotations, following somewhat faithfully
the Glozz_ model for annotations.

This is low-level in the sense that we make little attempt to interpret the
information stored in these annotations. For example, a relation might claim to
link two units of id unit42 and unit43. This being a low-level representation,
we simply note the fact. A higher-level representation might attempt to
actually make the corresponding units available to you, or perhaps provide
some sort of graph representation of them

.. _Glozz: http://erickow.com/posts/anno-models-glozz.html
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=too-many-arguments, protected-access
# pylint: disable=too-few-public-methods

from itertools import chain


class Span(object):
    """
    What portion of text an annotation corresponds to.
    Assumed to be in terms of character offsets

    The way we interpret spans in educe amounts to how Python
    interprets array slice indices.

    One way to understand them is to think of offsets as
    sitting in between individual characters ::

          h   o   w   d   y
        0   1   2   3   4   5

    So `(0,5)` covers the whole word above, and `(1,2)`
    picks out the letter "o"
    """
    def __init__(self, start, end):
        self.char_start = start
        self.char_end = end

    def __str__(self):
        return '(%d,%d)' % (self.char_start, self.char_end)

    def __repr__(self):
        return 'Span(%d, %d)' % (self.char_start, self.char_end)

    def __lt__(self, other):
        return self.char_start < other.char_start or\
            (self.char_start == other.char_start and
             self.char_end < other.char_end)

    def __eq__(self, other):
        return\
            self.char_start == other.char_start and\
            self.char_end == other.char_end

    def __gt__(self, other):
        return other < self

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return other <= self

    def __hash__(self):
        return (self.char_start, self.char_end).__hash__()

    def length(self):
        """
        Return the length of this span
        """
        return self.char_end - self.char_start

    def shift(self, offset):
        """
        Return a copy of this span, shifted to the right
        (if offset is positive) or left (if negative).

        It may be a bit more convenient to use 'absolute/relative'
        if you're trying to work with spans that are within other
        spans.
        """
        return Span(self.char_start + offset, self.char_end + offset)

    def absolute(self, other):
        """
        Assuming this span is relative to some other span,
        return a suitably shifted "absolute" copy.
        """
        return self.shift(other.char_start)

    def relative(self, other):
        """
        Assuming this span is relative to some other span,
        return a suitably shifted "absolute" copy.
        """
        return self.shift(0 - other.char_start)

    def encloses(self, other):
        """
        Return True if this span includes the argument

        Note that `x.encloses(x) == True`

        Corner case: `x.encloses(None) == False`

        See also `educe.graph.EnclosureGraph` if you might be repeating
        these checks
        """
        if other is None:
            return False
        else:
            return\
                self.char_start <= other.char_start and\
                self.char_end >= other.char_end

    def overlaps(self, other, inclusive=False):
        """
        Return the overlapping region if two spans have regions
        in common, or else None. ::

            Span(5, 10).overlaps(Span(8, 12)) == Span(8, 10)
            Span(5, 10).overlaps(Span(11, 12)) == None

        If `inclusive == True`, spans with touching edges are
        considered to overlap ::

            Span(5, 10).overlaps(Span(10, 12)) == None
            Span(5, 10).overlaps(Span(10, 12), inclusive=True) == Span(10, 10)

        """
        if other is None:
            return None
        elif self.encloses(other):
            return other
        elif other.encloses(self):
            return self
        else:
            common_start = max(self.char_start, other.char_start)
            common_end = min(self.char_end, other.char_end)
            if inclusive and common_start <= common_end:
                return Span(common_start, common_end)
            if common_start < common_end:
                return Span(common_start, common_end)
            else:
                return None

    def merge(self, other):
        """
        Return a span that stretches from the beginning to the
        end of the two spans. Whereas `overlaps` can be thought of
        as returning the intersection of two spans, this can be
        thought of as returning the union.
        """
        big_start = min(self.char_start, other.char_start)
        big_end = max(self.char_end, other.char_end)
        return Span(big_start, big_end)

    @classmethod
    def merge_all(cls, spans):
        """
        Return a span that stretches from the beginning to the end
        of all the spans in the list
        """
        spans = list(spans)
        if len(spans) < 1:
            raise ValueError("must have at least one span")
        big_start = min(x.char_start for x in spans)
        big_end = max(x.char_end for x in spans)
        return Span(big_start, big_end)


# pylint: disable=invalid-name
class RelSpan(object):
    """
    Which two units a relation connects.
    """
    def __init__(self, t1, t2):
        self.t1 = t1
        "string: id of an annotation"

        self.t2 = t2
        "string: id of an annotation"

    def __str__(self):
        return '%s -> %s' % (self.t1, self.t2)

    def __repr__(self):
        return 'RelSpan(%s, %s)' % (self.t1, self.t2)
# pylint: enable=invalid-name


# pylint: disable=no-self-use
class Standoff(object):
    """A standoff object ultimately points to some piece of text.

    The pointing is not necessarily direct though.

    Attributes
    ----------
    origin : educe.corpus.FileId, optional
        FileId of the document supporting this standoff.
    """
    def __init__(self, origin=None):
        self.origin = origin

    def _members(self):
        """Any annotations contained within this annotation.

        Must return None if is a terminal annotation (not the same
        meaning as returning the empty list).
        Non-terminal annotations must override this.

        Returns
        -------
        res : list of Standoff or None
            Annotations contained within this annotation ; None for
            terminal annotations.
        """
        return None

    def _terminals(self, seen=None):
        """Terminal annotations contained within this annotation.

        For terminal annotations, this is just the annotation itself.
        For non-terminal annotations, this recursively fetches the
        terminals.

        Parameters
        ----------
        seen : optional
            List of already annotations that have already been seen, so
            as to avoid returning duplicates.

        Returns
        -------
        res : list of Standoff
            List of terminal annotations for this annotation.
        """
        my_members = self._members()
        if my_members is None:
            return [self]
        seen = seen or []
        return chain.from_iterable([m._terminals(seen=seen + my_members)
                                    for m in my_members if m not in seen])

    def text_span(self):
        """
        Return the span from the earliest terminal annotation contained here
        to the latest.

        Corner case: if this is an empty non-terminal (which would be a very
        weird thing indeed), return None.

        Returns
        -------
        res : Span or None
            Span from the first character of the earliest terminal
            annotation contained here, to the last character of the
            latest terminal annotation ; None if this annotation has no
            terminal.
        """
        terminals = list(self._terminals())
        if len(terminals) > 0:
            start = min(t.span.char_start for t in terminals)
            end = max(t.span.char_end for t in terminals)
            return Span(start, end)
        else:
            return None

    def encloses(self, other):
        """
        True if this annotation's span encloses the span of the other.

        `s1.encloses(s2)` is shorthand for
        `s1.text_span().encloses(s2.text_span())`

        Parameters
        ----------
        other : Standoff
            Other annotation.

        Returns
        -------
        res : boolean
            True if this annotation's span encloses the span of the
            other.
        """
        return self.text_span().encloses(other.text_span())

    def overlaps(self, other):
        """
        True if this annotations's span overlaps with the span of the other.

        `s1.overlaps(s2)` is shorthand for
        `s1.text_span().overlaps(s2.text_span())`

        Parameters
        ----------
        other : Standoff
            Other annotation.

        Returns
        -------
        res : boolean
            True if this annotation's span overlaps with the span of the
            other.
        """
        return self.text_span().overlaps(other.text_span())
# pylint: enable=no-self-use


class Annotation(Standoff):
    """Any sort of annotation.

    Annotations tend to have:
    * span:     some sort of location (what they are annotating)
    * type:     some key label (we call a type)
    * features: an attribute to value dictionary
    """
    def __init__(self, anno_id, span, atype, features, metadata=None,
                 origin=None):
        """Init method.

        Parameters
        ----------
        anno_id : TODO
            Identifier for this annotation.
        span : Span
            Coordinates of the annotated span.
        atype : str
            Annotation type.
        features : dict from str to str
            Feature as a dict from feature_name to feature_value.
        metadata : dict from str to str, optional
            Metadata for the annotation, eg. author, creation date...
        origin : FileId, optional
            FileId of the document that supports this annotation.
        """
        Standoff.__init__(self, origin)
        self.origin = origin
        self._anno_id = anno_id
        self.span = span
        self.type = atype
        self.features = features
        self.metadata = metadata

    def __lt__(self, other):
        return self._anno_id < other._anno_id

    def __str__(self):
        feats = str(self.features)
        return ('%s [%s] %s %s' %
                (self.identifier(), self.type, self.span, feats))

    def local_id(self):
        """Local identifier.

        An identifier which is sufficient to pick out this annotation
        within a single annotation file.
        """
        return self._anno_id

    def identifier(self):
        """Global identifier if possible, else local identifier.

        String representation of an identifier that should be unique
        to this corpus at least.

        If the unit has an origin (see "FileId"), we use the

        * document
        * subdocument
        * stage
        * (but not the annotator!)
        * and the id from the XML file

        If we don't have an origin we fall back to just the id provided
        by the XML file.

        See also `position` as potentially a safer alternative to this
        (and what we mean by safer)
        """
        local_id = self._anno_id
        if self.origin is None:
            return local_id
        else:
            return self.origin.mk_global_id(local_id)


class Unit(Annotation):
    """Unit annotation.

    An annotation over a span of text.

    """

    def __init__(self, unit_id, span, utype, features, metadata=None,
                 origin=None):
        Annotation.__init__(self, unit_id, span, utype, features,
                            metadata, origin)

    def position(self):
        """
        The position is the set of "geographical" information only to identify
        an item. So instead of relying on some sort of name, we might rely on
        its text span. We assume that some name-based elements (document name,
        subdocument name, stage) can double as being positional.

        If the unit has an origin (see "FileId"), we use the

        * document
        * subdocument
        * stage
        * (but not the annotator!)
        * and its text span

        **position vs identifier**

        This is a trade-off.
        On the one hand, you can see the position as being a safer way
        to identify a unit, because it obviates having to worry about
        your naming mechanism guaranteeing stability across the board
        (eg. two annotators stick an annotation in the same place; does
        it have the same name).
        On the *other* hand, it's a bit harder to uniquely identify
        objects that may coincidentally fall in the same span.
        So how much do you trust your IDs?
        """
        if self.origin is None:
            ostuff = []
        else:
            ostuff = [self.origin.doc,
                      self.origin.subdoc,
                      self.origin.stage]
            span = self.span
        return ":".join(ostuff + [str(span.char_start), str(span.char_end)])


class Relation(Annotation):
    """An annotation between two annotations.

    Relations are directed; see `RelSpan` for details

    Use the `source` and `target` field to grab these respective
    annotations, but note that they are only instantiated after
    `fleshout` is called (corpus slurping normally fleshes out
    documents and thus their relations).

    """

    def __init__(self, rel_id, span, rtype, features, metadata=None):
        """Init method.

        Parameters
        ----------
        rel_id : string
            Relation id
        span : RelSpan
            Pair of units connected by this relation
        rtype : string
            Relation type
        features : dict
            Features
        metadata : dict from str to str, optional
            Metadata for this annotation.
        """
        Annotation.__init__(self, rel_id, span, rtype, features, metadata)
        self.source = None  # to be defined in fleshout
        'source annotation; will be defined by fleshout'

        self.target = None
        'target annotation; will be defined by fleshout'

    def _members(self):
        return [self.source, self.target]

    def fleshout(self, objects):
        """
        Given a dictionary mapping ids to annotation objects, set this
        relation's source and target fields.
        """
        source_span = self.span.t1
        target_span = self.span.t2
        if source_span not in objects:
            oops = 'There is no annotation with id %s [relation source]' %\
                source_span
            raise Exception(oops)
        elif target_span not in objects:
            oops = 'There is no annotation with id %s [relation target]' %\
                target_span
            raise Exception(oops)
        else:
            self.source = objects[source_span]
            self.target = objects[target_span]


class Schema(Annotation):
    """
    An annotation between a set of annotations

    Use the `members` field to grab the annotations themselves.
    But note that it is only created when `fleshout` is called.

    :type units: set(string)
    :type relations: set(string)
    :type schemas: set(string)
    """
    def __init__(self, rel_id, units, relations, schemas, stype,
                 features, metadata=None):
        self.units = units
        self.relations = relations
        self.schemas = schemas
        member_ids = units | relations | schemas
        self.members = None  # to be defined :-/
        Annotation.__init__(self, rel_id, member_ids, stype,
                            features, metadata)

    def terminals(self):
        """
        All unit-level annotations contained in this schema or
        (recursively in schema contained herein)
        """
        return list(self._terminals())

    def _members(self):
        return self.members

    def fleshout(self, objects):
        """
        Given a dictionary mapping ids to annotation objects, set this
        schema's `members` field to point to the appropriate objects
        """
        self.members = []
        for i in self.span:
            if i not in objects:
                oops = 'There is no annotation with id %s [schema member]' % i
                raise Exception(oops)
            self.members.append(objects[i])


class Document(Standoff):
    """
    A single (sub)-document.

    This can be seen as collections of unit, relation, and schema annotations
    """
    def __init__(self, units, relations, schemas, text):
        Standoff.__init__(self, None)

        self.units = units
        self.relations = relations
        self.schemas = schemas
        objects = {}

        for i in self.units:
            objects[i.local_id()] = i
        for i in self.relations:
            objects[i.local_id()] = i
        for i in self.schemas:
            objects[i.local_id()] = i

        for anno in self.relations:
            anno.fleshout(objects)
        for anno in self.schemas:
            anno.fleshout(objects)

        self._text = text

    def annotations(self):
        """
        All annotations associated with this document
        """
        return self.units + self.relations + self.schemas

    def _members(self):
        return self.annotations()

    def fleshout(self, origin):
        """
        See `set_origin`
        """
        self.set_origin(origin)

    def set_origin(self, origin):
        """
        If you have more than one document, it's a good idea to
        set its origin to a file ID so that you
        can more reliably the annotations apart.

        :type origin: :py:class:`educe.corpus.FileId`
        """
        self.origin = origin
        for anno in self.annotations():
            anno.origin = origin

    def global_id(self, local_id):
        """
        String representation of an identifier that should be unique
        to this corpus at least.
        """
        if self.origin is None:
            return local_id
        else:
            return self.origin.mk_global_id(local_id)

    def text(self, span=None):
        """
        Return the text associated with these annotations (or None),
        optionally limited to a span
        """
        if self._text is None:
            return None
        elif span is None:
            return self._text
        else:
            return self._text[span.char_start:span.char_end]
