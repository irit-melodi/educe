# Author: Eric Kow
# License: BSD3

"""
Low-level representation of corpus annotations.

This is low-level in the sense that we make little attempt to interpret the
information stored in these annotations. For example, a relation might claim to
link two units of id unit42 and unit43. This being a low-level representation,
we simply note the fact. A higher-level representation might attempt to
actually make the corresponding units available to you, or perhaps provide
some sort of graph representation of them
"""

class Span:
    """
    What portion of text an annotation corresponds to.
    Assumed to be in terms of character offsets
    """
    def __init__(self, start, end):
        self.char_start=start
        self.char_end=end

    def  __str__(self):
        return ('(%d,%d)' % (self.char_start, self.char_end))

class RelSpan():
    """
    Which two units a relation connections.
    """
    def __init__(self, t1, t2):
        self.t1=t1
        self.t2=t2

    def  __str__(self):
        return ('%s -> %s' % (self.t1, self.t2))

class Annotation:
    """
    Any sort of annotation. Annotations tend to have

    * span:     some sort of location (what they are annotating)
    * type:     some key label (we call a type)
    * features: a set of attribute value-pairs
    """
    def __init__(self, anno_id, span, type, features, origin=None):
        self.origin=origin
        self.__anno_id=anno_id
        self.span=span
        self.type=type
        self.features=features

    def __str__(self):
        feats=", ".join(map(str,self.features))
        return ('%s [%s] %s %s' % (self.identifier(),self.type, self.span, feats))

    def identifier(self):
        """
        String representation of an identifier that should be unique
        to this corpus at least.

        If the unit has an origin (see "FileId"), we use the

        * document
        * subdocument
        * stage
        * (but not the annotator!)
        * and the id from the XML file

        If we don't have an origin we fall back to just the id provided
        by the XML file

        See also `position` as potentially a safer alternative to this
        (and what we mean by safer)
        """
        o=self.origin
        if o is None:
            ostuff=[]
        else:
            ostuff=[o.doc, o.subdoc, o.stage]
        return ":".join(ostuff + [self.__anno_id])

class Unit(Annotation):
    """
    An annotation over a span of text
    """
    def __init__(self, unit_id, span, type, features, origin=None):
        Annotation.__init__(self, unit_id, span, type, features, origin)

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

        This is a trade-off.  One the hand, you can see the position as being
        a safer way to identify a unit, because it obviates having to worry
        about your naming mechanism guaranteeing stability across the board
        (eg. two annotators stick an annotation in the same place; does it have
        the same name). On the *other* hand, it's a bit harder to uniquely
        identify objects that may coincidentally fall in the same span.  So
        how much do you trust your IDs?
        """
        o=self.origin
        if o is None:
            ostuff=[]
        else:
            ostuff=[o.doc, o.subdoc, o.stage]
        return ":".join(ostuff + map(str,[self.span.char_start, self.span.char_end]))

class Relation(Annotation):
    """
    An annotation between two units.
    Relations are directed; see `RelSpan` for details
    """
    def __init__(self, rel_id, span, type, features):
        Annotation.__init__(self, rel_id, span, type, features)

class Feature():
    """Attribute-value pair"""
    def __init__(self, attribute, value):
        self.attribute=attribute
        self.value=value

    def __str__(self):
        if self.value is None:
            return self.attribute
        else:
            return ('%s:%s' % (self.attribute,self.value))

class Document:
    """
    A single (sub)-document.

    This can be seen as collections of unit and relation annotations
    """
    def __init__(self, units, rels):
        self.units=units
        self.rels=rels
