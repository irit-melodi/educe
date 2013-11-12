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

ex_attribution0="""Wr, Comm, Null, Null
also, Expansion.Conjunction"""
###
ex_attribution1="""Ot, Comm, Null, Null
9..35
0,0;0,1,0;0,1,2;0,2
#### Text ####
CenTrust Savings Bank said
##############
"""

class PdtbParseTest(unittest.TestCase):

    def assertOneResult(self, res):
        self.assertEqual(len(res),1)

    def test_text(self):
        res  = p._RawText.parseString(ex_txt)
        expected = 'federal thrift\n\nregulators ordered it to suspend \n\n####\n\ndividend payments on its two classes of preferred stock  '
        self.assertOneResult(res)
        self.assertEqual(expected, res[0])

    def test_nat(self):
        expected = 3
        txt      = str(expected)
        res      = p._nat.parseString(txt)
        self.assertOneResult(res)
        self.assertEqual(expected, res[0])

    def test_gorn(self):
        expected = p.GornAddress([0,1,5,3])
        txt      = ','.join(map(str,expected.parts))
        res = p._gorn.parseString(txt)
        self.assertEqual(len(res), 1)
        self.assertEqual(expected, res[0])

    def test_selection(self):
        res  = p._selection.parseString(ex_selection)
        self.assertOneResult(res)
        res0 = res.pop()
        self.assertTrue(isinstance(res0, p.Selection))
        expected = p.Selection(span=[(36,139)],
                               gorn=[p.GornAddress([0,1,1]),p.GornAddress([2,1])],
                               text='federal thrift regulators ordered it to suspend dividend payments on its two classes of preferred stock')
        self.assertEqual(expected.text, res0.text)
        self.assertEqual(expected, res0)

    def test(self):
        for path in glob.glob('tests/*.pdtb'):
            try:
                xs = p.parse(path)
            except pp.ParseException as e:
                doc = open(path).read()
                xs  = p._relationList.parseString(doc)
                for x in xs:
                    print >> sys.stderr
                    print >> sys.stderr, x
                print >> sys.stderr, path
                raise e
