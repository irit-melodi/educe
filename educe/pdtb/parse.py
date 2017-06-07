#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author Eric Kow
# LICENSE: BSD3 (2013, Université Paul Sabatier)

"""
Standalone parser for PDTB files.

The function `parse` takes a single .pdtb file and returns a list
of `Relation`, with the following subtypes:

+--------------------+-----------------+------------------+------+
| Relation           | selection       | features         | sup? |
+====================+=================+==================+======+
| `ExplicitRelation` | `Selection`     | attr, 1 connhead | Y    |
+--------------------+-----------------+------------------+------+
| `ImplicitRelation` | `InferenceSite` | attr, 2 conn     | Y    |
+--------------------+-----------------+------------------+------+
| `AltLexRelation`   | `Selection`     | attr, 2 semclass | Y    |
+--------------------+-----------------+------------------+------+
| `EntityRelation`   | `InferenceSite` | none             | N    |
+--------------------+-----------------+------------------+------+
| `NoRelation`       | `InferenceSite` | none             | N    |
+--------------------+-----------------+------------------+------+

These relation subtypes are stitched together (and inherit members) from
two or three components

    * arguments: always `arg1` and `arg2`; but in some cases, the
      arguments can have supplementary information
    * selection: see either `Selection` or `InferenceSite`
    * some features (see eg. `ExplictRelationFeatures`)

The simplest way to get to grips with this may be to try the `parse`
function on some sample relations and print the resulting objects.
"""

import codecs
import re
import sys

if sys.version > '3':
    from functools import reduce
    from io import StringIO
else:
    from StringIO import StringIO

import funcparserlib.parser as fp


# ---------------------------------------------------------------------
# parse results
# ---------------------------------------------------------------------
class PdtbItem(object):
    @classmethod
    def _prefered_order(cls):
        """
        Preferred order for printing key/value pairs
        """
        return ['text',
                'sentnum', 'strpos', 'span', 'gorn',
                'semclass', 'connective', 'connective1', 'connective2',
                'attribution',
                'arg1', 'arg2']

    def _substr(self):
        d = self.__dict__
        ks1 = [k for k in self._prefered_order() if k in d]
        ks2 = [k for k in d if k not in self._prefered_order()]
        return '\n '.join('%s = %s' % (k, d[k]) for k in ks1 + ks2)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, self._substr())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        # thanks, http://stackoverflow.com/a/390511/446326
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other


class GornAddress(PdtbItem):
    def __init__(self, parts):
        self.parts = parts

    def __str__(self):
        return '.'.join(str(x) for x in self.parts)


class Attribution(PdtbItem):
    def __init__(self, source, type, polarity, determinacy, selection=None):
        self.source = source
        self.type = type
        self.polarity = polarity
        self.determinacy = determinacy
        self.selection = selection

    def _substr(self):
        selStr = '@ %s' % self.selection._substr() if self.selection else ''
        return '%s %s %s %s%s' %\
            (self.source, self.type, self.polarity, self.determinacy, selStr)


class InferenceSite(PdtbItem):
    def __init__(self, strpos, sentnum):
        self.strpos = strpos
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
    def __init__(self, text, semclass1, semclass2=None):
        self.text = text
        assert isinstance(semclass1, SemClass)
        if semclass2:
            assert isinstance(semclass2, SemClass)
        self.semclass1 = semclass1
        self.semclass2 = semclass2

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
        if attribution:
            assert isinstance(attribution, Attribution)
        if sup:
            assert isinstance(sup, Sup)
        self.attribution = attribution
        self.sup = sup

    def _substr(self):
        sup_str = ' + %s' % self.sup if self.sup else ''
        return '%s | %s%s' % (
            Selection._substr(self), self.attribution, sup_str)


class Relation(PdtbItem):
    """
    Attributes
    ----------
    arg1 : TODO
        TODO
    arg2 : TODO
        TODO
    """
    def __init__(self, args):
        if len(args) == 4:
            sup1, arg1, arg2, sup2 = args
            self.arg1 = Arg(arg1, arg1.attribution, sup1) if sup1 else arg1
            self.arg2 = Arg(arg2, arg2.attribution, sup2) if sup2 else arg2
        elif len(args) == 2:
            self.arg1, self.arg2 = args
        else:
            raise ValueError('Was expecting either 2 or 4 arguments, '
                             'but got: %d\n%s' % (len(args), args))

    def _substr(self):
        return PdtbItem._substr(self)


class ExplicitRelationFeatures(PdtbItem):
    def __init__(self, attribution, connhead):
        assert isinstance(attribution, Attribution)
        assert isinstance(connhead, Connective)
        self.attribution = attribution
        self.connhead = connhead

    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.attribution, other.connhead)


class ImplicitRelationFeatures(PdtbItem):
    def __init__(self, attribution, connective1, connective2=None):
        assert isinstance(attribution, Attribution)
        assert isinstance(connective1, Connective)
        if connective2:
            assert isinstance(connective2, Connective)
        self.attribution = attribution
        self.connective1 = connective1
        self.connective2 = connective2

    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.attribution,
                     other.connective1, other.connective2)


class AltLexRelationFeatures(PdtbItem):
    def __init__(self, attribution, semclass1, semclass2):
        assert isinstance(attribution, Attribution)
        assert isinstance(semclass1, SemClass)
        if semclass2:
            assert isinstance(semclass2, SemClass)
        self.attribution = attribution
        self.semclass1 = semclass1
        self.semclass2 = semclass2

    @classmethod
    def _init_copy(cls, self, other):
        cls.__init__(self, other.attribution, other.semclass1, other.semclass2)


class ExplicitRelation(Selection, ExplicitRelationFeatures, Relation):
    def __init__(self, selection, features, args):
        Relation.__init__(self, args)
        Selection._init_copy(self, selection)
        ExplicitRelationFeatures._init_copy(self, features)

    def _substr(self):
        return Relation._substr(self)


class ImplicitRelation(InferenceSite, ImplicitRelationFeatures, Relation):
    def __init__(self, infsite, features, args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)
        ImplicitRelationFeatures._init_copy(self, features)

    def _substr(self):
        return Relation._substr(self)


class AltLexRelation(Selection, AltLexRelationFeatures, Relation):
    def __init__(self, selection, features, args):
        Relation.__init__(self, args)
        Selection._init_copy(self, selection)
        AltLexRelationFeatures._init_copy(self, features)

    def _substr(self):
        return Relation._substr(self)


class EntityRelation(InferenceSite, Relation):
    def __init__(self, infsite, args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)

    def _substr(self):
        return Relation._substr(self)


class NoRelation(InferenceSite, Relation):
    def __init__(self, infsite, args):
        Relation.__init__(self, args)
        InferenceSite._init_copy(self, infsite)

    def _substr(self):
        return Relation._substr(self)


# ---------------------------------------------------------------------
# not-quite-lexing
# ---------------------------------------------------------------------
#
# FIXME
# funcparserlib works on a stream of arbitrary tokens, eg. the output of
# a lexer. At the time of this writing, I didn't trust any of the fancy
# tokenisation libraries because I was suspicious of them messing up the
# whitespace (we have natural language text in here); but now on second
# thought maybe I'm being dumb. If we could have a lexer that blocks out
# the raw text bits eg. `r'#### Text ####\n(.*?)\n##############'`; and
# provide some abstractions over tokens, we could maybe simplify the
# parser a lot... which could in turn make it faster?
#
class _Char(object):
    def __init__(self, value, abspos, line, relpos):
        self.value = value
        self.abspos = abspos
        self.line = line
        self.relpos = relpos

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __repr__(self):
        char = self.value
        if self.value == '\n':
            char = 'NL'
        elif self.value == ' ':
            char = 'SP'
        elif self.value == '\t':
            char = 'TAB'
        return '[%s] %d (line: %d col: %d)' % (
            char, self.abspos, self.line, self.relpos)


def _annotate_production(s):
    return s


def _annotate_debug(s):
    """
    Add line/col char number
    """
    def tokens():
        line = 1
        col = 1
        pos = 1
        for c in StringIO(s).read():
            yield _Char(c, pos, line, col)
            pos += 1
            if c == '\n':
                line += 1
                col = 1
            else:
                col += 1
    return list(tokens())


# ---------------------------------------------------------------------
# funcparserlib utilities
# ---------------------------------------------------------------------
_DEBUG = 0  # turn this on to get line number hints
_const = lambda x: lambda _: x
_unarg = lambda f: lambda x: f(*x)


def _cons(pair):
    head, tail = pair
    return [head] + tail


def _mkstr_debug(x):
    return "".join(c.value for c in x)


def _mkstr_production(x):
    return "".join(x)


_any = fp.some(_const(True))


def _intersperse(d, xs):
    """
    a -> [a] -> [a]
    """
    xs2 = []
    if xs:
        xs2.append(xs[0])
    for x in xs[1:]:
        xs2.append(d)
        xs2.append(x)
    return xs2


def _not_followed_by(p):
    """Parser(a, b) -> Parser(a, b)

    Without actually consuming any tokens, succeed if the parser would fail
    """

    @fp.Parser
    def _helper(tokens, s):
        res = []
        try:
            p.run(tokens, s)
        except fp.NoParseError as e:
            return fp._Ignored(()), s
        raise fp.NoParseError(u'followed by something we did not want', s)

    _helper.name = u'not_followed_by{ %s }' % p.name
    return _helper


def _skipto(p):
    """Parser(a, b) -> Parser(a, [a])

    Returns a parser that returns all tokens parsed until the given
    parser succeeds (we assume here you want to skip the end parser)
    """

    @fp.Parser
    def _helper(tokens, s):
        """Iterative implementation preventing the stack overflow."""
        res = []
        s2 = s
        while s2.pos < len(tokens):
            try:
                (v, s3) = p.run(tokens, s2)
                return res, s3
            except fp.NoParseError as e:
                res.append(tokens[s2.pos])
                pos = s2.pos + 1
                s2 = fp.State(pos, max(pos, s2.max))
        raise fp.NoParseError(u'no tokens left in the stream', s)

    _helper.name = u'{ skip_to %s }' % p.name
    return _helper


def _skipto_mkstr(p):
    return _skipto(p) >> _mkstr


def _satisfies_debug(fn):
    return fp.some(lambda t: fn(t.value))


def _satisfies_production(fn):
    return fp.some(fn)


def _oneof(xs):
    return _satisfies(lambda x: x in xs)


def _sepby(delim, p):
    return p + fp.many(fp.skip(delim) + p) >> _cons


def _sequence(ps):
    return reduce(lambda x, y: x + y, ps)


def _many_char(fn):
    return fp.many(_satisfies(fn)) >> _mkstr


def _noise(xs):
    """String -> Parser(a, ())

    Skip over this literal string
    """
    @fp.Parser
    def _helper(tokens, s):
        """Iterative implementation preventing the stack overflow."""
        res = []
        start = s.pos
        end = start + len(xs)
        toks = tokens[start:end]
        if _DEBUG:
            vals = [t.value for t in toks]
        else:
            vals = toks
        if vals == xs:
            pos = s.pos + len(xs)
            s2 = fp.State(pos, max(pos, s.max))
            return fp._Ignored(()), s2
        else:
            raise fp.NoParseError(u'Did not match literal ' + xs, s)

    _helper.name = u'{ literal %s }' % xs
    return _helper


if _DEBUG:
    _annotate = _annotate_debug
    _mkstr = _mkstr_debug
    _satisfies = _satisfies_debug
else:
    _annotate = _annotate_production
    _mkstr = _mkstr_production
    _satisfies = _satisfies_production


# ---------------------------------------------------------------------
# elementary parts
# ---------------------------------------------------------------------
_nat = fp.oneplus(_satisfies(lambda c: c.isdigit())) >> (
    lambda x: int(_mkstr(x)))
_nl = fp.skip(_oneof("\r\n"))
_comma = fp.skip(_oneof(","))
_semicolon = fp.skip(_oneof(";"))
_fullstop = fp.skip(_oneof("."))
# horizontal only
_sp = fp.skip(_many_char(lambda x: x not in "\r\n" and x.isspace()))
_allsp = fp.skip(_many_char(lambda x: x.isspace()))
_alphanum_str = _many_char(lambda x: x.isalnum())
_eof = fp.skip(fp.finished)


class _OptionalBlock:
    """
    For use with `_lines` only: wraps a parser so that we not
    only take in account that it's optional but that one of
    the newlines around it is optional too

    `avoid` is used in case of possible ambiguity; it lets us
    stop parsing if we hit an alternative (better) interpretation
    """
    def __init__(self, p, avoid=None):
        self.avoid = avoid
        self.p = p


def _words(ps):
    """
    Ignore horizontal whitespace between elements
    """
    return _sequence(_intersperse(_sp, ps))


def _lines(ps):
    if not ps:
        raise Exception('_lines must be called with at least one parser')
    elif isinstance(ps[0], _OptionalBlock):
        raise Exception('Sorry, first block cannot be optional')

    def _prefix_nl(y):
        return _nl + y

    def _next(y, prefix=_prefix_nl):
        if isinstance(y, _OptionalBlock):
            if y.avoid:
                # stop parsing if we see the distractor
                distractor = prefix(y.avoid)
                p_next = _not_followed_by(distractor) + prefix(y.p)
            else:
                p_next = prefix(y.p)
            return fp.maybe(p_next)
        else:
            return prefix(y)

    def _combine(x, y):
        return x + _next(y)

    return reduce(_combine, ps)


def _section_begin(t):
    return _noise('____' + t + '____')


def _subsection_begin(t):
    return _noise('#### ' + t + ' ####')


_subsection_end = _noise('##############')
_bar = _noise('_' * 56)

_span = _nat + _noise('..') + _nat >> tuple
_gorn = _sepby(_comma, _nat) >> GornAddress
_StringPosition = _nat
_SentenceNumber = _nat


# ---------------------------------------------------------------------
# selections - funcparserlib
# ---------------------------------------------------------------------
_SpanList = _sepby(_semicolon, _span)
_GornAddressList = _sepby(_semicolon, _gorn)
_RawText = _lines([_subsection_begin('Text'),
                   _skipto_mkstr(_nl + _subsection_end)])

_selection =\
        _lines([_SpanList, _GornAddressList, _RawText]) >> _unarg(Selection)

_inferenceSite =\
        _lines([_StringPosition, _SentenceNumber]) >> _unarg(InferenceSite)


# ---------------------------------------------------------------------
# features
# ---------------------------------------------------------------------
_Source = _alphanum_str
_Type = _alphanum_str
_Polarity = _alphanum_str
_Determinacy = _alphanum_str

_attributionCoreFeatures =\
        _words(_intersperse(_comma,
                            [_Source, _Type, _Polarity, _Determinacy]))

_attributionFeatures =\
        _lines([_subsection_begin('Features'),
                _attributionCoreFeatures,
                _OptionalBlock(_selection)]) >> _unarg(Attribution)

# Expansion.Alternative.Chosen alternative =>
# Expansion / Alternative / "Chosen alternative "
_SemanticClassWord = _many_char(lambda x: x in [' ', '-'] or x.isalnum())
_SemanticClassN = _sepby(_fullstop, _SemanticClassWord) >> SemClass
_SemanticClass1 = _SemanticClassN
_SemanticClass2 = _SemanticClassN
_semanticClass = _SemanticClass1 + fp.maybe(_sp + _comma + _sp +
                                            _SemanticClass2)

# always followed by a comma (yeah, a bit clunky)
_ConnHead = _skipto_mkstr(_comma)
_Conn1 = _ConnHead
_Conn2 = _ConnHead


def _mkConnective(c, semclasses):
    return Connective(c, *semclasses)


_connHeadSemanticClass = _ConnHead + _sp + _semanticClass >> _unarg(
    _mkConnective)
_conn1SemanticClass = _Conn1 + _sp + _semanticClass >> _unarg(
    _mkConnective)
_conn2SemanticClass = _Conn2 + _sp + _semanticClass >> _unarg(
    _mkConnective)


# ---------------------------------------------------------------------
# arguments and supplementary information
# ---------------------------------------------------------------------
def _Arg(name):
    return _section_begin(name.capitalize())


def _Sup(name):
    return _section_begin(name.capitalize())


def _arg(name):
    p = _lines([_Arg(name), _selection, _attributionFeatures]) >> _unarg(Arg)
    return p


def _arg_no_features(name):
    p = _lines([_Arg(name), _selection]) >> Arg
    return p


def _sup(name):
    p = _lines([_Sup(name), _selection]) >> Sup
    return p


# this is a bit yucky because I don't really know how to express
# optional first blocks and make sure I handle the intervening
# newlines correctly
def _mk_args_and_sups():
    rest = [_arg('arg1'),
            _arg('arg2'),
            _OptionalBlock(_sup('sup2'))]

    with_sup1 = _lines([_sup('sup1')] + rest) >> tuple
    sans_sup1 = _lines(rest) >> (lambda xs: tuple([None] + list(xs)))
    return with_sup1 | sans_sup1  # yuck :-(


_args_and_sups = _mk_args_and_sups()
_args_only =\
        _lines([_arg_no_features('arg1'),
                _arg_no_features('arg2')]) >> tuple


# ---------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------
__Explicit = 'Explicit'
__Implict = 'Implicit'
__AltLex = 'AltLex'
__EntRel = 'EntRel'
__NoRel = 'NoRel'

_Explicit = _section_begin(__Explicit)
_Implict = _section_begin(__Implict)
_AltLex = _section_begin(__AltLex)
_EntRel = _section_begin(__EntRel)
_NoRel = _section_begin(__NoRel)

_explicitRelationFeatures =\
        _lines([_attributionFeatures, _connHeadSemanticClass])\
        >> _unarg(ExplicitRelationFeatures)

_altLexRelationFeatures =\
        _lines([_attributionFeatures, _semanticClass])\
        >> (lambda x: AltLexRelationFeatures(x[0], *x[1]))

_afterImplicitRelationFeatures =\
        _section_begin('Arg1') | _section_begin('Sup1')

_implicitRelationFeatures =\
        _lines([_attributionFeatures,
                _conn1SemanticClass,
                _OptionalBlock(_conn2SemanticClass,
                               avoid=_afterImplicitRelationFeatures)])\
        >> _unarg(ImplicitRelationFeatures)

_explicitRelation =\
        _lines([_selection, _explicitRelationFeatures, _args_and_sups])\
        >> _unarg(ExplicitRelation)

_altLexRelation =\
        _lines([_selection, _altLexRelationFeatures, _args_and_sups])\
        >> _unarg(AltLexRelation)

_implicitRelation =\
        _lines([_inferenceSite, _implicitRelationFeatures, _args_and_sups])\
        >> _unarg(ImplicitRelation)

_entityRelation =\
        _lines([_inferenceSite, _args_only])\
        >> _unarg(EntityRelation)

_noRelation =\
        _lines([_inferenceSite, _args_only])\
        >> _unarg(NoRelation)

_relationParts = [
    (__Explicit, _explicitRelation),
    (__Implict, _implicitRelation),
    (__AltLex, _altLexRelation),
    (__EntRel, _entityRelation),
    (__NoRel, _noRelation),
]


def _relationBody(ty, core):
    return _lines([_section_begin(ty), core])


def _orRels(rs):
    """
    R1 or R2 or .. RN
    """
    cores = [_relationBody(*r) for r in rs]
    return _lines([_bar,
                   reduce(lambda x, y: x | y, cores),
                   _bar])


def _oneRel(ty, core):
    return _lines([_bar, _relationBody(ty, core), _bar])


_relation = _orRels(_relationParts)

_relationList = _sepby(_nl, _relation)
_pdtbRelation = _relation + _allsp + _eof
_pdtbFile = _relationList + _allsp + _eof


# ---------------------------------------------------------------------
# tests and examples
# ---------------------------------------------------------------------
def split_relations(s):
    frame = r'________________________________________________________\n' +\
            r'.*?' +\
            r'________________________________________________________'
    return re.findall(frame, s, re.DOTALL)


def parse_relation(s):
    """
    Parse a single relation or throw a ParseException.
    """
    type_re = r'^________________________________________________________\n' +\
              r'____(?P<type>.*)____\n'
    rtype = re.match(type_re, s).group('type')
    rules = dict(_relationParts)
    if rtype not in rules:
        raise Exception('Unknown PDTB relation type: ' + rtype)
    parser = _oneRel(rtype, rules[rtype]) + _eof
    return parser.parse(_annotate(s))


def parse(path):
    """Retrieve the list of relations found in a single .pdtb file.

    Parameters
    ----------
    path : str
        Path to the .pdtb file (?)

    Returns
    -------
    relations : list of Relation
        List of relations found.
    """
    doc = codecs.open(path, 'r', 'iso8859-1').read()
    return _pdtbFile.parse(_annotate(doc))
    # alternatively: using a regular expression to split into relations
    # and parsing each relation separately - perhaps more robust?
    # splits  = split_relations(doc)
    # return [ parse_relation(s) for s in splits  ]
