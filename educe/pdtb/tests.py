import glob
import sys
import unittest

import educe.pdtb.parse as p
import pyparsing as pp

ex_txt="""#### Text ####
federal thrift

regulators ordered it to suspend 

####

dividend payments on its two classes of preferred stock  
##############"""

ex_selection="""36..139
0,1,1;2,1
#### Text ####
federal thrift regulators ordered it to suspend dividend payments on its two classes of preferred stock
##############"""


ex_implicit_attribution="""#### Features ####
Wr, Comm, Null, Null
also, Expansion.Conjunction"""
###

ex_implicit_features="""#### Features ####
Wr, Comm, Null, Null
in particular, Expansion.Restatement.Specification
because, Contingency.Cause.Reason"""

ex_attribution1="""#### Features ####
Ot, Comm, Null, Null"""

ex_attribution2="""#### Features ####
Ot, Comm, Null, Null
9..35
0,0;0,1,0;0,1,2;0,2
#### Text ####
CenTrust Savings Bank said
##############"""

ex_sup1="""____Sup1____
1730..1799
11,2,3
#### Text ####
blop blop split shares
##############"""

ex_implicit_rel="""
"""

ex_frame="""________________________________________________________
blah blah bla
_____tahueoa______
bop
________________________________________________________"""


class PdtbParseTest(unittest.TestCase):

    def assertParse(self, parser, expected, txt):
        parser   = parser + p._eof
        res      = parser.parse(p._annotate(txt))
        self.assertEqual(expected, res)

    def test_skipto(self):
        expected = 'blah blah blah hooyeah'
        txt      = expected + ','
        self.assertParse(p._skipto_mkstr(p._comma), expected, txt)

    def test_lines(self):
        expected = 'abc'
        txt      = 'a\nb\nc'
        char     = lambda x:p._oneof(x)
        parser   = p._lines([char("a"), char("b"), char("c")]) >> p._mkstr
        self.assertParse(parser, expected, txt)

        parser   = p._lines([char("a"), char("b"), p._OptionalBlock(char("c"))]) >> p._mkstr
        self.assertParse(parser, expected, txt)

    def test_tok(self):
        expected = [p._Char('h',1,1,1),
                    p._Char('i',2,1,2),
                    p._Char('\n',3,1,3),
                    p._Char('y',4,2,1),
                    p._Char('o',5,2,2),
                    p._Char('u',6,2,3),
                    ]
        tokens = list(p._annotate_debug('hi\nyou'))
        self.assertEqual(expected, tokens)

    def test_nat(self):
        expected = 42
        txt      = str(expected)
        self.assertParse(p._nat, expected, txt)

    def test_span(self):
        expected = (8,12)
        txt      = '8..12'
        self.assertParse(p._span, expected, txt)

    def test_gorn(self):
        expected = p.GornAddress([0,1,5,3])
        txt      = ','.join(map(str,expected.parts))
        self.assertParse(p._gorn, expected, txt)

    def test_span_list(self):
        expected = [(8,12),(9,3),(10,39)]
        txt      = '8..12;9..3;10..39'
        self.assertParse(p._SpanList, expected, txt)

    def test_text(self):
        expected = 'federal thrift\n\nregulators ordered it to suspend \n\n####\n\ndividend payments on its two classes of preferred stock  '
        txt      = ex_txt
        self.assertParse(p._RawText, expected, txt)

    def test_selection(self):
        expected = p.Selection(span=[(36,139)],
                               gorn=[p.GornAddress([0,1,1]),p.GornAddress([2,1])],
                               text='federal thrift regulators ordered it to suspend dividend payments on its two classes of preferred stock')
        txt      = ex_selection
        self.assertParse(p._selection, expected, txt)

    def test_attribution_core(self):
        expected = ('Wr', 'Comm', 'Null', 'Null')
        txt      = "Wr, Comm, Null, Null"
        self.assertParse(p._attributionCoreFeatures, expected, txt)

    def test_attribution(self):
        expected = p.Attribution('Ot', 'Comm', 'Null', 'Null')
        txt      = ex_attribution1
        self.assertParse(p._attributionFeatures, expected, txt)

    def test_attribution_sel(self):
        expected_sel = p.Selection(span=[(9,35)],
                                   gorn=[p.GornAddress([0,0]),
                                         p.GornAddress([0,1,0]),
                                         p.GornAddress([0,1,2]),
                                         p.GornAddress([0,2])],
                                   text='CenTrust Savings Bank said')
        expected = p.Attribution('Ot', 'Comm', 'Null', 'Null', expected_sel)
        txt      = ex_attribution2
        self.assertParse(p._attributionFeatures, expected, txt)

    def test_semclass(self):
        expected = 'Chosen alternative'
        txt      = expected
        self.assertParse(p._SemanticClassWord, expected, txt)

        expected1 = p.SemClass(['Expansion', 'Alternative', 'Chosen alternative'])
        expected  = expected1
        txt       = 'Expansion.Alternative.Chosen alternative'
        self.assertParse(p._SemanticClass1, expected, txt)

        expected  = (expected1, None)
        self.assertParse(p._semanticClass, expected, txt)

        expected2 = p.SemClass(['Contingency', 'Cause', 'Result'])
        expected  = (expected1, expected2)
        txt       = 'Expansion.Alternative.Chosen alternative, Contingency.Cause.Result'
        self.assertParse(p._semanticClass, expected, txt)

    def test_connective(self):
        expected = p.Connective('also', p.SemClass(['Expansion','Conjunction']))
        txt      = 'also, Expansion.Conjunction'
        self.assertParse(p._conn1SemanticClass, expected, txt)
        self.assertParse(p._conn2SemanticClass, expected, txt)

    def test_sup(self):
        expected_sel = p.Selection(span=[(1730,1799)],
                                   gorn=[p.GornAddress([11,2,3])],
                                   text='blop blop split shares')
        expected = p.Sup(expected_sel)
        txt      = ex_sup1
        self.assertParse(p._sup('sup1'), expected, txt)

    def test_implicit_features_1(self):
        expected_attr = p.Attribution('Wr', 'Comm', 'Null', 'Null')
        expected_conn = p.Connective('also', p.SemClass(['Expansion','Conjunction']))
        expected = p.ImplicitRelationFeatures(expected_attr, expected_conn, None)
        txt      = ex_implicit_attribution
        self.assertParse(p._implicitRelationFeatures, expected, txt)

    def test_implicit_features_2(self):
        expected_conn1 = p.Connective('in particular',
                                      p.SemClass(['Expansion','Restatement','Specification']))
        expected_conn2 = p.Connective('because',
                                      p.SemClass(['Contingency','Cause','Reason']))
        expected_attr  = p.Attribution('Wr', 'Comm', 'Null', 'Null')
        expected = p.ImplicitRelationFeatures(expected_attr, expected_conn1, expected_conn2)
        txt      = ex_implicit_features
        self.assertParse(p._implicitRelationFeatures, expected, txt)

    def test_frame(self):
        expected = [ex_frame]
        split    = p.split_relations(ex_frame)
        self.assertEqual(expected, split)

    def test(self):
        for path in glob.glob('tests/*.pdtb'):
            try:
                xs = p.parse(path)
                self.assertNotEquals(0, len(xs))
            except pp.ParseException as e:
                doc = open(path).read()
                xs  = p._relationList.parseString(doc)
                for x in xs:
                    print >> sys.stderr
                    print >> sys.stderr, x
                print >> sys.stderr, path
                raise e
