#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author Eric Kow
# LICENSE: BSD3 (2013, Université Paul Sabatier)

"""
Standalone parser for PDTB files.

The function `parse` takes a single .pdtb file and returns a list
of `Relation`, with the following subtypes:

    * `ExplicitRelation`
    * `ImplicitRelation`
    * `AltLexRelation`
    * `EntityRelation`
    * `AltLexRelation`

Note that aside from having two arguments, these do not have very
much to do with each other, but there is certainly some overlap.
"""

import copy
import pyparsing as pp

# ---------------------------------------------------------------------
# parse results
# ---------------------------------------------------------------------

class PdtbItem(object):
    @classmethod
    def _prefered_order(self):
        """
        Preferred order for printing key/value pairs
        """
        return []

    def _substr(self):
        d   = self.__dict__
        ks1 = [ k for k in self._prefered_order() if k in d     ]
        ks2 = [ k for k in d if k not in self._prefered_order() ]
        return '\n '.join('%s = %s' % (k,d[k]) for k in ks1 + ks2)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, self._substr())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # thanks, http://stackoverflow.com/a/390511/446326
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

class GornAddress(PdtbItem):
    def __init__(self, parts):
        self.parts = parts

    def __str__(self):
        return '.'.join(map(str,self.parts))

class Attribution(PdtbItem):
    def __init__(self, source, type, polarity, determinacy, selection=None):
        self.source      = source
        self.type        = type
        self.polarity    = polarity
        self.determinacy = determinacy
        self.selection   = selection

    def _substr(self):
        selStr = '@ %s' % self.selection._substr() if self.selection else ''
        return '%s %s %s %s%s' %\
            (self.source, self.type, self.polarity, self.determinacy, selStr)

class InferenceSite(PdtbItem):
    def __init__(self, strpos, sentnum):
        self.strpos  = strpos
        self.sentnum = sentnum

    def _substr(self):
        return '%d [sent %d]' % (self.strpos, self.sentnum)

    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.strpos, other.sentnum)

class Selection(PdtbItem):
    def __init__(self, span, gorn, text):
        self.span = span
        self.gorn = gorn
        self.text = text

    def _substr(self):
        return '%s %s %s' % (self.span, self.gorn, self.text)

    # FIXME: is there an Pythonic to achieve something of this sort,
    # where we'd like to initialise a subclass from a class instance
    # by copying all its fields?
    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.span, other.gorn, other.text)

class Connective(PdtbItem):
    def __init__(self, text, semclasses):
        self.text = text
        self.semclass1   = semclasses[0]
        if len(semclasses) > 1:
            self.semclass2 = semclasses[1]
        else:
            self.semclass2 = None

    def _substr(self):
        fields = [self.text, self.semclass1._substr()]
        if self.semclass2:
            fields.append(self.semclass2._substr())
        return ' | '.join(fields)

class SemClass(PdtbItem):
    def __init__(self, klass):
        self.klass = klass

    def _substr(self):
        return '.'.join(self.klass)

class Sup(Selection):
    def __init__(self, selection):
        Selection._init_copy(self, selection)

class Arg(Selection):
    def __init__(self, selection, attribution=None, sup=None):
        Selection._init_copy(self, selection)
        self.attribution = attribution
        self.sup         = sup

    def _substr(self):
        sup_str = ' + %s' % self.sup if self.sup else ''
        return '%s | %s%s' % (Selection._substr(self), self.attribution, sup_str)

class Relation(PdtbItem):
    def __init__(self, args):
        xs = list(args)
        if isinstance(xs[-1], Sup):
            sup2 = xs.pop()
        else:
            sup2 = None

        arg2 = xs.pop()
        arg1 = xs.pop()
        if xs and isinstance(xs[-1], Sup):
            sup1 = xs.pop()
        else:
            sup1 = None

        self.arg1 = Arg(arg1, arg1.attribution, sup1)
        self.arg2 = Arg(arg2, arg2.attribution, sup2)

    def _prefered_order(cls):
        return ['text',
                'sentnum', 'strpos', 'span', 'gorn',
                'semclass', 'connective', 'connective1', 'connective2',
                'attribution',
                'arg1', 'arg2']

    def _substr(self):
        return PdtbItem._substr(self)

class ExplicitRelation(Selection, Relation):
    def __init__(self, selection, attribution, connective, *args):
        # FIXME: if I use super(Relation,self).__init__(args),
        #
        # I get an error I don't quite understand
        #    site-packages/pyparsing-1.5.6-py2.7.egg/pyparsing.py", line 675, in wrapper
        #    return func(*args[limit[0]:])
        # TypeError: <lambda>() takes exactly 1 argument (0 given)
        Relation.__init__(self, args)
        Selection._init_copy(self, selection)
        self.attribution = attribution
        self.connective  = connective

    def _substr(self):
        return Relation._substr(self)

class ImplicitRelationFeatures(PdtbItem):
    def __init__(self, attribution, connective1, connective2=None):
        self.attribution = attribution
        self.connective1 = connective1
        self.connective2 = connective2

    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.attribution,
                     other.connective1, other.connective2)

class ImplicitRelation(InferenceSite, ImplicitRelationFeatures, Relation):
    def __init__(self, infsite, features, *args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)
        ImplicitRelationFeatures._init_copy(self, features)

    def _substr(self):
         return Relation._substr(self)

class AltLexRelation(Selection, Relation):
    def __init__(self, selection, attribution, semclass, *args):
        Relation.__init__(self, args)
        Selection._init_copy(self, selection)
        self.attribution = attribution
        self.semclass    = semclass

    def _substr(self):
         return Relation._substr(self)

class EntityRelation(InferenceSite, Relation):
    def __init__(self, infsite, *args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)

    def _substr(self):
         return Relation._substr(self)

class NoRelation(InferenceSite, Relation):
    def __init__(self, infsite, *args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)

    def _substr(self):
         return Relation._substr(self)

# ---------------------------------------------------------------------
# elementary parts
# ---------------------------------------------------------------------

# note that this tries to hew to the grammar in the
# PDTB Annotation Manual 2 (Section 6.3: File Format)
#
# apologies if it's confusing that we distiguish between
# upper case (terminal symbols) vs lower case rules,
# as per the manual

# configure pyparsing not to skip over newlines
p_DWC = [ x for x in pp.ParseElementEnhance.DEFAULT_WHITE_CHARS if x != "\n" ]
pp.ParseElementEnhance.setDefaultWhitespaceChars("".join(p_DWC))

def _list(p, delim=';'):
    p = pp.Group(pp.delimitedList(p, delim=delim))
    # FIXME: magic code
    # If `p.parseString(foo) :: ParseResult(a)`, what I want is
    # `_list(p).parseString(bar) :: ParseResult([a])`
    #
    # and not some nested craziness like
    # `ParseResult([ParseResult(a)])`
    #
    # So I'm not entirely sure I understand my solution yet,
    # because there is something baffling to me about pyparsing
    #
    # I understand vaguely that ParseResult implements `__iter__`,
    # but for some reason given `ts :: ParseResult([a])`,
    # `ts[0] :: a` (and not `[a]`?!)
    p.setParseAction(lambda ts:[list(t) for t in ts])
    return p

def _noise(t):
    return pp.Suppress(pp.Literal(t)).setName("「%s」" % t)

def _avoiding(distractor, p):
    """
    For use in cases where `p` would try to parse something that
    would be better understood as `distractor`.

    Stop parsing if we see `distractor`
    """
    return pp.FollowedBy(distractor) ^ (~ distractor + p)

_nl           = pp.Suppress(pp.LineEnd()).setName('NL')
_alphanum_str = pp.Word(pp.alphanums).setName('alphaNum')
_comma        = _noise(',').setName(',')
_nat          = pp.Word(pp.nums).setParseAction(lambda t: int(t[0])).setName('natural')

class _OptionalBlock:
    """
    For use with `_lines` only: wraps a parser so that we not
    only take in account that it's optional but that one of
    the newlines around it is optional too
    """
    def __init__(self, p, avoid=None):
        self.avoid = avoid
        self.p     = p

def _lines(ps):
    """
    First block cannot be Optional!
    """
    assert not isinstance(ps[0], _OptionalBlock)
    def _combine(x,y):
        if isinstance(y,_OptionalBlock):
            nl_y   = pp.Optional(_nl + y.p)
            p_next = _avoiding(_nl + y.avoid, nl_y) if y.avoid else nl_y
            return x + p_next
        else:
            return x + _nl + y
    return reduce(_combine, ps)

def _section_begin(t):
    return _noise('____' + t + '____')

def _subsection_begin(t):
    return _noise('#### ' + t + ' ####')

def _act(f):
    """
    Helper to call constructors in a somewhat point-free way
    """
    return lambda ts:f(*ts)

_subsection_end = _noise('##############')
_bar            = _noise('_' * 56)

_span = (_nat + _noise('..') + _nat).setName('span')
_gorn = _list(_nat, delim=',').setName('gorn')
_span.setParseAction(tuple)
_gorn.setParseAction(lambda t:GornAddress(list(t[0])))
_StringPosition = _nat.copy()
_SentenceNumber = _nat.copy()

# ---------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------

_SpanList        = _list(_span).setName('spanList')
_GornAddressList = _list(_gorn).setName('gornList')
_RawText = _lines([_subsection_begin('Text'),
                   pp.SkipTo(_nl + _subsection_end, include=True)])

_selection =\
        _lines([_SpanList, _GornAddressList, _RawText])
_selection.setParseAction(_act(Selection))

_inferenceSite = _lines([_StringPosition, _SentenceNumber])
_inferenceSite.setParseAction(_act(InferenceSite))

# ---------------------------------------------------------------------
# features
# ---------------------------------------------------------------------

_Source      = _alphanum_str.copy().setName('Source')
_Type        = _alphanum_str.copy().setName('Type')
_Polarity    = _alphanum_str.copy().setName('Polarity')
_Determinacy = _alphanum_str.copy().setName('Determinacy')

_attributionCoreFeatures =\
        (_Source   + _comma +\
         _Type     + _comma +\
         _Polarity + _comma +\
         _Determinacy)

_attributionFeatures =\
        _lines([_subsection_begin('Features'),
                _attributionCoreFeatures,
                _OptionalBlock(_selection)])
_attributionFeatures.setParseAction(_act(Attribution))

# Expansion.Alternative.Chosen alternative =>
# Expansion / Alternative / "Chosen alternative "
_SemanticClassWord = pp.Word(pp.alphanums + ' -')
_SemanticClassN = pp.Group(pp.delimitedList(_SemanticClassWord, delim='.'))
_SemanticClassN.setParseAction(_act(SemClass))
_SemanticClassN.setName('semanticClass')
_SemanticClass1 = _SemanticClassN.copy()
_SemanticClass2 = _SemanticClassN.copy()
_semanticClass  = pp.Group(_SemanticClass1 + pp.Optional(_comma + _SemanticClass2))

# always followed by a comma (yeah, a bit clunky)
_ConnHead = pp.SkipTo(_comma, include=True)
_Conn1    = _ConnHead.copy()
_Conn2    = _ConnHead.copy()

_connHeadSemanticClass = _ConnHead + _semanticClass
_connHeadSemanticClass.setParseAction(_act(Connective))

_conn1SemanticClass = _Conn1 + _semanticClass
_conn1SemanticClass.setParseAction(_act(Connective))

_conn2SemanticClass = _Conn2 + _semanticClass
_conn2SemanticClass.setParseAction(_act(Connective))

# ---------------------------------------------------------------------
# arguments and supplementary information
# ---------------------------------------------------------------------

def _Arg(name):
    return _section_begin(name.capitalize())

def _Sup(name):
    return _section_begin(name.capitalize())

def _arg(name):
    p = _lines([_Arg(name), _selection, _attributionFeatures])
    p.setParseAction(_act(Arg))
    return p

def _arg_no_features(name):
    p = _lines([_Arg(name), _selection])
    p.setParseAction(_act(Arg))
    return p

def _sup(name):
    p = _lines([_Sup(name), _selection])
    p.setParseAction(_act(Sup))
    return p

def _specRelation(xs,mini=False):
    args_full = [_OptionalBlock(_sup('sup1')),
                 _arg('arg1'),
                 _arg('arg2'),
                _OptionalBlock(_sup('sup2'))]
    args_mini = [_arg_no_features('arg1'),
                 _arg_no_features('arg2')]
    args = args_mini if mini else args_full
    return _lines(xs + args)

# ---------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------

_Explicit = _section_begin('Explicit')
_Implict  = _section_begin('Implicit')
_AltLex   = _section_begin('AltLex')
_EntRel   = _section_begin('EntRel')
_NoRel    = _section_begin('NoRel')


_explicitRelationFeatures =\
        _lines([_attributionFeatures,
                _connHeadSemanticClass])

_altLexRelationFeatures =\
        _lines([_attributionFeatures, _semanticClass])

_implicitRelationFeatures =\
        _lines([_attributionFeatures,
                _conn1SemanticClass,
                _OptionalBlock(_conn2SemanticClass,
                               avoid=_section_begin('Arg1'))])
_implicitRelationFeatures.setParseAction(_act(ImplicitRelationFeatures))

_explicitRelation =\
        _specRelation([_selection, _explicitRelationFeatures]).\
        setParseAction(_act(ExplicitRelation))

_altLexRelation =\
        _specRelation([_selection, _altLexRelationFeatures]).\
        setParseAction(_act(AltLexRelation))

_implicitRelation =\
        _specRelation([_inferenceSite, _implicitRelationFeatures]).\
        setParseAction(_act(ImplicitRelation))

_entityRelation =\
        _specRelation([_inferenceSite], mini=True).\
        setParseAction(_act(EntityRelation))

_noRelation =\
        _specRelation([_inferenceSite], mini=True).\
        setParseAction(_act(NoRelation))

def _orRels(rs):
    return _lines([_bar,
                   reduce(lambda x, y: x ^ y, [_lines(r) for r in rs]),
                   _bar])

_relation =\
        _orRels([(_Explicit, _explicitRelation),
                 (_Implict,  _implicitRelation),
                 (_AltLex,   _altLexRelation),
                 (_EntRel,   _entityRelation),
                 (_NoRel,    _noRelation),
                 ])

_relationList = _list(_relation, delim=_nl)

_pdtbFile = _relationList + pp.Suppress(pp.White() + pp.StringEnd())

# ---------------------------------------------------------------------
# tests and examples
# ---------------------------------------------------------------------

def parse(path):
    """
    Parse a single .pdtb file and return the list of relations found
    within

    :rtype: [Relation]
    """
    doc     = open(path).read()
    return _pdtbFile.parseString(doc)[0]
