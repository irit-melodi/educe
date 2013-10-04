import codecs
import glob
import sys

from educe.rst_dt import parse

# ---------------------------------------------------------------------
# example tree snippets
# ---------------------------------------------------------------------

tstr0="""
( Root (span 5 6)
  ( Satellite (leaf 5) (rel2par act:goal) (text <EDU>x</EDU>) )
  ( Nucleus   (leaf 6) (rel2par act:goal) (text <EDU>y</EDU>) )
)
"""

tstr1="""
( Root (span 1 9)
  ( Nucleus (leaf 1) (rel2par textualOrganization) (text <s><EDU> ORGANIZING YOUR MATERIALS </EDU></s>) )
  ( Satellite (span 2 9) (rel2par textualOrganization)
    ( Satellite (span 2 4) (rel2par general:specific)
      ( Nucleus (span 2 3) (rel2par preparation:act)
        ( Satellite (leaf 2) (rel2par preparation:act) (text <s><EDU> Once you've decided on the kind of paneling you want to install --- and the pattern ---</EDU>) )
        ( Nucleus (leaf 3) (rel2par preparation:act) (text <EDU>some preliminary steps remain</EDU>) )
      )
      ( Satellite (leaf 4) (rel2par preparation:act) (text <EDU>before you climb into your working clothes. </EDU></s>) )
    )
    ( Nucleus (span 5 9) (rel2par general:specific)
      ( Nucleus (span 5 8) (rel2par preparation:act)
        ( Nucleus (span 5 7) (rel2par step1:step2)
          ( Nucleus (span 5 6) (rel2par preparation:act)
            ( Satellite (leaf 5) (rel2par act:goal) (text <s><EDU> You'll need to measure the wall or room to be paneled,</EDU>) )
            ( Nucleus (leaf 6) (rel2par act:goal) (text <EDU>estimate the amount of paneling you'll need,</EDU>) )
          )
          ( Nucleus (leaf 7) (rel2par preparation:act) (text <EDU>buy the paneling,</EDU>) )
        )
        ( Nucleus (leaf 8) (rel2par step1:step2) (text <EDU>gather the necessary tools and equipment (see illustration on page 87),</EDU>) )
      )
      ( Nucleus (leaf 9) (rel2par preparation:act) (text <EDU>and even condition certain types of paneling before installation. </EDU></s>) )
    )
  )
)
"""

text1="\n".join(\
        [" ORGANIZING YOUR MATERIALS ",
         " Once you've decided on the kind of paneling you want to install --- and the pattern ---",
         "some preliminary steps remain",
         "before you climb into your working clothes. ",
         " You'll need to measure the wall or room to be paneled,",
         "estimate the amount of paneling you'll need,",
         "buy the paneling,",
         "gather the necessary tools and equipment (see illustration on page 87),",
         "and even condition certain types of paneling before installation. "
         ])

tstr2="""
( Root (span 1 6)
  ( Nucleus (span 1 6) (rel2par span)
    ( Satellite (leaf 1) (rel2par evaluation-s) (text _!The back of the Moth.<P>_!) )
    ( Nucleus (span 2 6) (rel2par span)
      ( Nucleus (span 2 4) (rel2par Same-Unit)
        ( Nucleus (leaf 2) (rel2par span) (text _!Baron Bromley III,_!) )))))
"""

text2="The back of the Moth.<P>\nBaron Bromley III,"

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

def test_tstr0():
    parse.RSTTree.build(tstr0)

def test_tstr1():
    t      = parse.RSTTree.build(tstr1)
    t_text = t.text()
    sp     = t.node.span
    assert t.edu_span()    == (1,9)
    assert t_text          == text1
    assert sp.char_end     == len(t_text)

def test_tstr2():
    t      = parse.RSTTree.build(tstr2)
    t_text = t.text()
    sp     = t.node.span
    assert t.edu_span()    == (1,6)
    assert t_text          == text2
    assert sp.char_end     == len(t_text)

def test_from_files():
    for i in glob.glob('tests/rst*.dis'):
        t      = parse.read_annotation_file(i)
        assert t.node.span.char_end == len(t.text())
