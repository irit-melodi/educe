from __future__ import print_function
import glob
import os
import random
import unittest
import copy

from educe.rst_dt import annotation, parse, SimpleRSTTree
from educe.rst_dt.dep2con import deptree_to_simple_rst_tree
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.parse import (parse_lightweight_tree,
                                parse_rst_dt_tree,
                                read_annotation_file)
from ..internalutil import treenode

# ---------------------------------------------------------------------
# example tree snippets
# ---------------------------------------------------------------------

TSTR0 = """
( Root (span 5 6)
  ( Satellite (leaf 5) (rel2par act:goal) (text <EDU>x</EDU>) )
  ( Nucleus   (leaf 6) (rel2par act:goal) (text <EDU>y</EDU>) )
)
"""

TSTR1 = """
( Root (span 1 9)
  ( Nucleus (leaf 1) (rel2par textualOrganization)
                     (text <s><EDU> ORGANIZING YOUR MATERIALS </EDU></s>) )
  ( Satellite (span 2 9) (rel2par textualOrganization)
    ( Satellite (span 2 4) (rel2par general:specific)
      ( Nucleus (span 2 3) (rel2par preparation:act)
        ( Satellite (leaf 2) (rel2par preparation:act)
          (text <s><EDU> Once you've decided on the kind of paneling you want to install --- and the pattern ---</EDU>) )
        ( Nucleus (leaf 3) (rel2par preparation:act)
          (text <EDU>some preliminary steps remain</EDU>) )
      )
      ( Satellite (leaf 4) (rel2par preparation:act)
          (text <EDU>before you climb into your working clothes. </EDU></s>) )
    )
    ( Nucleus (span 5 9) (rel2par general:specific)
      ( Nucleus (span 5 8) (rel2par preparation:act)
        ( Nucleus (span 5 7) (rel2par step1:step2)
          ( Nucleus (span 5 6) (rel2par preparation:act)
            ( Satellite (leaf 5) (rel2par act:goal)
                (text <s><EDU> You'll need to measure the wall or room to be paneled,</EDU>) )
            ( Nucleus (leaf 6) (rel2par act:goal)
                (text <EDU>estimate the amount of paneling you'll need,</EDU>) )
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

TEXT1 = " ".join([
    " ORGANIZING YOUR MATERIALS ",
    (" Once you've decided on the kind of paneling you want to install "
     "--- and the pattern ---"),

    "some preliminary steps remain",
    "before you climb into your working clothes. ",
    " You'll need to measure the wall or room to be paneled,",
    "estimate the amount of paneling you'll need,",
    "buy the paneling,",
    ("gather the necessary tools and equipment (see illustration "
     "on page 87),"),

    "and even condition certain types of paneling before installation. "
])


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


class RSTTest(unittest.TestCase):
    _trees = {}  # will be lazily populated
    longMessage = True

    def test_tstr0(self):
        parse_rst_dt_tree(TSTR0)

    def test_tstr1(self):
        t = parse_rst_dt_tree(TSTR1)
        t_text = t.text()
        sp = treenode(t).span
        self.assertEqual((1, 9), t.edu_span())
        self.assertEqual(TEXT1, t_text)
        self.assertEqual(len(t_text), sp.char_end)

    def test_from_files(self):
        for i in glob.glob('tests/*.dis'):
            t = read_annotation_file(i, os.path.splitext(i)[0])
            self.assertEqual(len(t.text()), treenode(t).span.char_end)

    def _test_trees(self):
        if not self._trees:
            self._trees = {}
            for i, tstr in enumerate([TSTR0, TSTR1]):
                self._trees["tstr%d" % i] = parse.parse_rst_dt_tree(tstr)
            for i in glob.glob('tests/*.dis'):
                bname = os.path.basename(i)
                tfile = os.path.splitext(i)[0]
                self._trees[bname] = parse.read_annotation_file(i, tfile)
        return self._trees

    def test_binarize(self):
        for _, tree in self._test_trees().items():
            bin_tree = annotation._binarize(tree)
            self.assertTrue(annotation.is_binary(bin_tree))

    def test_rst_to_dt(self):
        lw_trees = [
            "(R:rel (S x) (N y))",

            """
            (R:rel
                (S x)
                (N:rel (N h) (S t)))
            """,

            """
            (R:r
                (S x)
                (N:r (N:r (S t1) (N h))
                     (S t2)))
            """
        ]

        for lstr in lw_trees:
            rst1 = parse_lightweight_tree(lstr)
            dep = RstDepTree.from_simple_rst_tree(rst1)
            rst2 = deptree_to_simple_rst_tree(dep)
            self.assertEqual(rst1, rst2, "round-trip on " + lstr)

        for name, tree in self._test_trees().items():
            rst1 = SimpleRSTTree.from_rst_tree(tree)
            dep = RstDepTree.from_simple_rst_tree(rst1)
            rst2 = deptree_to_simple_rst_tree(dep)
            self.assertEqual(treenode(rst1).span,
                             treenode(rst2).span,
                             "span equality on " + name)
            self.assertEqual(treenode(rst1).edu_span,
                             treenode(rst2).edu_span,
                             "edu span equality on " + name)

    def test_dt_to_rst_order(self):
        lw_trees = [
            "(R:r (N:r (N h) (S r1)) (S r2))",
            "(R:r (S:r (S l2) (N l1)) (N h))",
            "(R:r (N:r (S l1) (N h)) (S r1))",
            """
            (R:r
              (N:r
                (N:r (S l2)
                     (N:r (S l1)
                          (N h)))
                (S r1))
              (S r2))
            """,  # ((l2 <- l1 <- h) -> r1 -> r2)
            """
            (R:r
              (N:r
                (S l2)
                (N:r (N:r (S l1)
                          (N h))
                     (S r1)))
              (S r2))
            """,  # (l2 <- ((l1 <- h) -> r1)) -> r2
        ]

        for lstr in lw_trees:
            rst1 = parse_lightweight_tree(lstr)
            dep = RstDepTree.from_simple_rst_tree(rst1)

            dep_a = dep
            rst2a = deptree_to_simple_rst_tree(dep_a)
            self.assertEqual(rst1, rst2a, "round-trip on " + lstr)

            dep_b = copy.deepcopy(dep)
            dep_b.deps(0).reverse()
            rst2b = deptree_to_simple_rst_tree(dep_b)
            # TODO assertion on rst2b?

            dep_c = copy.deepcopy(dep)
            random.shuffle(dep_c.deps(0))
            rst2c = deptree_to_simple_rst_tree(dep_c)
            # TODO assertion on rst2c?

    def test_rst_to_dt_nuclearity_loss(self):
        """
        Test that we still get sane tree structure with
        nuclearity loss
        """
        tricky = """
                 (R:r (S t) (N h))
                 """

        nuked = """
                (R:r (N t) (N h))
                """

#        tricky = """
#                 (R:r
#                     (S x)
#                     (N:r (N:r (S t1) (N h))
#                          (S t2)))
#                 """
#
#        nuked = """
#                 (R:r
#                     (N x)
#                     (N:r (N:r (N t1) (N h))
#                          (N t2)))
#                 """

        rst0 = parse_lightweight_tree(nuked)
        rst1 = parse_lightweight_tree(tricky)

        # a little sanity check first
        dep0 = RstDepTree.from_simple_rst_tree(rst0)
        rev0 = deptree_to_simple_rst_tree(dep0)  # was:, ['r'])
        self.assertEqual(rst0, rev0, "same structure " + nuked)  # sanity

        # now the real test
        dep1 = RstDepTree.from_simple_rst_tree(rst1)
        rev1 = deptree_to_simple_rst_tree(dep1)  # was:, ['r'])
        # self.assertEqual(rst0, rev1, "same structure " + tricky)
        # TODO restore a meaningful assertion
