import codecs
import glob
import sys

from educe.rst_dt import parse

def test_preprocess():
    tstr, minedu, maxedu = parse.preprocess(parse.test)
    assert minedu == 1
    assert maxedu == 9

def test_builtin():
    parse.RSTTree(parse.test0)
    t = parse.RSTTree(parse.test)
    leaf_text = [ l._text for l in t.leaves() ]
    assert parse.test_text == leaf_text

def test_from_files():
    for i in glob.glob('tests/rst*.dis'):
        parse.read_annotation_file(i)
