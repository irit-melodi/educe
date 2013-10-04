import codecs
import glob
import sys

from educe.rst_dt import parse

def test_builtin():
    parse.RSTTree.build(parse.test0)
    t      = parse.RSTTree.build(parse.test)
    t_text = t.text()
    sp     = t.node.span
    assert t.edu_span()    == (1,9)
    assert parse.test_text == t_text
    assert sp.char_end     == len(t_text)

def test_from_files():
    for i in glob.glob('tests/rst*.dis'):
        t      = parse.read_annotation_file(i)
        assert t.node.span.char_end == len(t.text())
