"""Conversion between dependency and constituency trees.

TODO
----
* [ ] refactor strategies
* [ ] use label+nuclearity to help determine the order of attachment:
      greedily take successive modifiers that belong to the same
      underlying multinuclear relation
"""

from collections import defaultdict, namedtuple
import itertools

from educe.annotation import Span
from educe.internalutil import treenode
from educe.rst_dt.annotation import (NUC_N, NUC_S, NUC_R, Node, RSTTree,
                                     SimpleRSTTree)
from educe.rst_dt.deptree import RstDtException


class DummyNuclearityClassifier(object):
    """Predict the nuclearity of each EDU using simple rules.

    Parameters
    ----------
    strategy : str
        Strategy to use to generate predictions.

        * "unamb_else_most_frequent": predicts multinuclear when the
        relation label is unambiguously multinuclear in the training
        set, mononuclear otherwise.
        * "most_frequent_by_rel": predicts the most frequent nuclearity
        for the given relation label in the training set.
        * "constant": always predicts a constant label provided by the
        user.

    constant : str
        The explicit constant as predicted by the "constant" strategy.
        This parameter is useful only for the "constant" strategy.

    TODO
    ----
    complete after `sklearn.dummy.DummyClassifier`
    """

    def __init__(self, strategy="unamb_else_most_frequent", constant=None):
        self.strategy = strategy
        self.constant = constant

    def fit(self, X, y):
        """Fit the dummy classifier.

        FIXME: currently a no-op.

        Both X and y are royally ignored.

        Parameters
        ----------
        X: list of RstDepTrees

        y: array-like, shape = [n_samples]
            Target nuclearity array for each EDU of each RstDepTree.
        """
        if self.strategy not in ("unamb_else_most_frequent",
                                 "most_frequent_by_rel",
                                 "constant"):
            raise ValueError("Unknown strategy type.")

        if (self.strategy == "constant" and
            self.constant not in (NUC_N, NUC_S)):
            # ensure that the constant value provided is acceptable
            raise ValueError("The constant target value must be "
                             "{} or {}".format(NUC_N, NUC_S))

        # special processing: ROOT is considered multinuclear
        # 2016-12-06 I'm unsure what form "root" should have at this
        # point, so all three possible values are currently included
        # but we should trim this list down (MM)
        multinuc_lbls = ['ROOT', 'root', '---']
        if self.strategy == "unamb_else_most_frequent":
            # FIXME automatically get these from the training set
            multinuc_lbls.extend(['joint', 'same-unit', 'textual'])

        elif self.strategy == "most_frequent_by_rel":
            # FIXME very dirty hack to avoid loading the RST-DT corpus
            # upfront (triggered by the import of
            # load_corpus_as_dataframe)
            try:
                from .corpus_diagnostics import (
                    get_most_frequent_unuc,
                    load_corpus_as_dataframe
                )
            except IOError:
                raise
            train_df = load_corpus_as_dataframe(selection='train')
            multinuc_lbls.extend(rel_name for rel_name, mode_unuc
                                 in get_most_frequent_unuc(train_df).items()
                                 if mode_unuc == 'NN')
        self.multinuc_lbls_ = multinuc_lbls

        # FIXME properly implement fit for the different strategies
        return self

    def predict(self, X):
        """Perform classification on test RstDepTrees X.
        """
        y = []
        for dtree in X:
            if self.strategy == "constant":
                yi = [self.constant for rel in dtree.labels]
                y.append(yi)
            else:
                # FIXME NUC_R for the root?
                # NB: we condition multinuclear relations on (i > head)
                yi = [(NUC_N if (i > head and rel in self.multinuc_lbls_)
                       else NUC_S)
                      for i, (head, rel)
                      in enumerate(itertools.izip(dtree.heads, dtree.labels))]
                y.append(yi)

        return y


class InsideOutAttachmentRanker(object):
    """Rank modifiers, from the inside out on either side.

    Given a dependency tree node and its children, return an order on
    the list of children, that should fulfill an inside-out traversal
    on either side.

    Let's back up a little bit for some background on this criterion.
    We assume that dependency tree nodes can be characterised and
    ordered by their position in the text. If we did such a thing, the
    head node would sit somewhere between its children ::

        lX, .., l2, l1, h, r1, r2, .., rY

    An inside-out traversal is one in which moves strictly outward
    from the centre node. Note that this does not imply any other
    requirement on the order our traversal (particularly on left vs
    right), so `h l1 l2 .. lX r1 r2, etc` is as much a valid
    inside-outside traversal as `h l1 r1 l2 etc`; however,
    `h l2 l1 etc` is right out.

    The ranking is determined by a strategy. Currently implemented
    strategies are:
    - 'id': stable sort that keeps the original interleaving, as
    defined below,
    - 'lllrrr': all left then right dependents,
    - 'rrrlll': all right then left dependents,
    - 'lrlrlr': alternating directions, left first,
    - 'rlrlrl': alternating directions, right first.

    """

    def __init__(self, strategy='id', prioritize_same_unit=False,
                 order='weak'):
        if strategy not in ['id',
                            'lllrrr', 'rrrlll',
                            'lrlrlr', 'rlrlrl',
                            'closest-lr', 'closest-rl',
                            'closest-intra-lr-inter-lr',
                            'closest-intra-rl-inter-rl',
                            'closest-intra-rl-inter-lr',
                            'sdist-edist-lr', 'sdist-edist-rl']:
            raise ValueError('Unknown transformation strategy '
                             '{stg}'.format(stg=strategy))
        self.strategy = strategy
        self.prioritize_same_unit = prioritize_same_unit
        if order not in ['weak', 'strict']:
            raise ValueError("Order must be one of {'weak', 'strict'}")
        self.order = order

    def fit(self, X, y):
        """Here a no-op."""
        return self

    def predict(self, X):
        """Predict order between modifiers of the same head.

        The predicted order should guarantee inside-out traversal on
        either side of a head.

        Parameters
        ----------
        X: list of RstDepTrees
            Dependency trees for which we want attachment rankings

        Returns
        -------
        dt_ranks: list of arrays of ranks
            Attachment ranking, one per RstDepTree

        Notes
        -----
        The '*intra*' strategies need to map each EDU to its surrounding
        sentence. This is currently done by attaching to the RstDepTree
        this mapping as a list, `sent_idx`.
        Future versions should make this right.
        """
        strategy = self.strategy

        if strategy == 'id':  # radical oracle: use gold rank as is
            # we know the true order, it is stored in dtree.ranks
            return [dtree.ranks for dtree in X]

        dt_ranks = []
        for dtree in X:
            # for each RstDepTree, the result will be an array of ranks
            ranks = [0 for hd in dtree.heads]  # initialize result

            unique_heads = set(dtree.heads[1:])  # exclude head of fake root
            for head in unique_heads:
                rank_idx = 1  # init rank

                targets = [i for i, hd in enumerate(dtree.heads)
                           if hd == head]
                if self.prioritize_same_unit:
                    # gobble everything between the head and the rightmost
                    # "same-unit"
                    # FIXME weak order: fragments of an n-ary "same-unit"
                    # should get the same order
                    same_unit_tgts = [tgt for tgt in targets
                                      if dtree.labels[tgt] == 'same-unit']
                    if same_unit_tgts:
                        # take first all dependents between the head
                        # and the rightmost same-unit
                        last_same_unit_tgt = same_unit_tgts[-1]
                        priority_tgts = [tgt for tgt in targets
                                         if (tgt > head and
                                             tgt <= last_same_unit_tgt)]
                        for tgt in priority_tgts:
                            ranks[tgt] = rank_idx
                            rank_idx += 1
                        # remove them from the remaining targets
                        targets = [tgt for tgt in targets
                                   if tgt not in priority_tgts]

                # elements to the left and right of the node respectively
                # these are stacks (outside ... inside)
                left = [i for i in targets if i < head]
                right = [i for i in targets if i > head]
                right = list(reversed(right))

                if strategy in ['lllrrr', 'rrrlll']:
                    # one side then the other
                    # FIXME weak order
                    sides = ([left, right] if strategy == 'lllrrr'
                             else [right, left])
                    for side in sides:
                        for tgt in side:
                            ranks[tgt] = rank_idx
                            rank_idx += 1

                elif strategy in ['lrlrlr', 'rlrlrl']:
                    # alternating sides
                    sides = ([left, right] if strategy == 'lrlrlr'
                             else [right, left])
                    while any(sides):
                        for side in sides:
                            if not side:
                                continue
                            dep_cur = side.pop()
                            ranks[dep_cur] = rank_idx
                            # weakly-ordered: consecutive nuclei with
                            # same label are assumed to be part of a
                            # multinuclear relation => same rank
                            lbl_cur = dtree.labels[dep_cur]
                            nuc_cur = dtree.nucs[dep_cur]
                            if self.order == 'weak' and nuc_cur == NUC_N:
                                while (side
                                       and dtree.labels[side[-1]] == lbl_cur
                                       and dtree.nucs[side[-1]] == nuc_cur):
                                    # give same rank
                                    dep_cur = side.pop()
                                    ranks[dep_cur] = rank_idx
                            # increment rank
                            rank_idx += 1

                elif strategy in ['closest-lr', 'closest-rl']:
                    # take closest dependents first, break ties using the
                    # side: lr to take left over right, rl to take right
                    # over left
                    sides = ([left, right] if strategy == 'closest-lr'
                             else [right, left])
                    while any(sides):
                        if left and right:
                            dist_left = abs(left[-1] - head)
                            dist_right = abs(right[-1] - head)
                            if dist_left == dist_right:
                                side = sides[0]
                            elif dist_left < dist_right:
                                side = left
                            else:
                                side = right
                        else:  # one side is empty
                            side = left if left else right
                        # same code as lrlrlr/rlrlrl above ; make into
                        # helper function?
                        dep_cur = side.pop()
                        ranks[dep_cur] = rank_idx
                        # weakly-ordered: consecutive nuclei with
                        # same label are assumed to be part of a
                        # multinuclear relation => same rank
                        lbl_cur = dtree.labels[dep_cur]
                        nuc_cur = dtree.nucs[dep_cur]
                        if self.order == 'weak' and nuc_cur == NUC_N:
                            while (side
                                   and dtree.labels[side[-1]] == lbl_cur
                                   and dtree.nucs[side[-1]] == nuc_cur):
                                # give same rank
                                dep_cur = side.pop()
                                ranks[dep_cur] = rank_idx
                        # increment rank
                        rank_idx += 1

                else:
                    # strategies that depend on intra/inter-sentential info
                    # NB: the way sentential info is stored is a dirty hack ;
                    # this should be fixed at some point
                    if not hasattr(dtree, 'sent_idx'):
                        raise ValueError(('Strategy {stg} depends on '
                                          'sentential information which is '
                                          'missing here'
                                          '').format(stg=strategy))

                    # sent_idx for all EDUs that need to be locally
                    # ranked (+ their head)
                    # FIXME write a clean imputation procedure
                    # that is global to all EDUs in the document
                    loc_edus = sorted(targets + [head])
                    sent_idc = [dtree.sent_idx[x] for x in loc_edus
                                if dtree.sent_idx[x] is not None]
                    if len(sent_idc) != len(loc_edus):
                        # missing sent_idx => (pseudo-)imputation ;
                        # this is a very local, and dirty, workaround
                        # * left dependents + head
                        sent_idc_left = []
                        sent_idx_cur = min(sent_idc) if sent_idc else 0
                        for x in loc_edus:
                            if x > head:
                                break
                            sent_idx_x = dtree.sent_idx[x]
                            if sent_idx_x is not None:
                                sent_idx_cur = sent_idx_x
                            sent_idc_left.append(sent_idx_cur)
                        # * right dependents
                        sent_idc_right = []
                        sent_idx_cur = max(sent_idc) if sent_idc else 0
                        for x in reversed(targets):
                            if x <= head:
                                break
                            sent_idx_x = dtree.sent_idx[x]
                            if sent_idx_x is not None:
                                sent_idx_cur = sent_idx_x
                            sent_idc_right.append(sent_idx_cur)
                        # * replace sent_idc with the result of the
                        # pseudo-imputation
                        sent_idc = (sent_idc_left +
                                    list(reversed(sent_idc_right)))
                    # build this into a dict
                    sent_idx_loc = {e: s_idx for e, s_idx
                                    in zip(loc_edus, sent_idc)}

                    # intra/inter strategies
                    if strategy in ['closest-intra-rl-inter-lr',
                                    'closest-intra-rl-inter-rl',
                                    'closest-intra-lr-inter-lr',
                                    'closest-intra-lr-inter-rl']:
                        # best: closest-intra-rl-inter-lr (2016-07-??)
                        # current: closest-intra-rl-inter-rl (2016-09-13)

                        # intra
                        left_intra = [tgt for tgt in left
                                      if sent_idx_loc[tgt] == sent_idx_loc[head]]
                        right_intra = [tgt for tgt in right
                                       if sent_idx_loc[tgt] == sent_idx_loc[head]]
                        sides = ([right_intra, left_intra]
                                 if strategy in ['closest-intra-rl-inter-lr',
                                                 'closest-intra-rl-inter-rl']
                                 else [left_intra, right_intra])
                        # same code as 'closest-*' above
                        while any(sides):
                            if left_intra and right_intra:
                                dist_left = abs(left_intra[-1] - head)
                                dist_right = abs(right_intra[-1] - head)
                                if dist_left == dist_right:
                                    side = sides[0]
                                elif dist_left < dist_right:
                                    side = left_intra
                                else:
                                    side = right_intra
                            else:  # one side is empty
                                side = (left_intra if left_intra
                                        else right_intra)
                            # same code as lrlrlr/rlrlrl above ; make into
                            # helper function?
                            dep_cur = side.pop()
                            ranks[dep_cur] = rank_idx
                            # weakly-ordered: consecutive nuclei with
                            # same label are assumed to be part of a
                            # multinuclear relation => same rank
                            lbl_cur = dtree.labels[dep_cur]
                            nuc_cur = dtree.nucs[dep_cur]
                            if self.order == 'weak' and nuc_cur == NUC_N:
                                while (side
                                       and dtree.labels[side[-1]] == lbl_cur
                                       and dtree.nucs[side[-1]] == nuc_cur):
                                    # give same rank
                                    dep_cur = side.pop()
                                    ranks[dep_cur] = rank_idx
                            # increment rank
                            rank_idx += 1

                        # inter
                        left_inter = [tgt for tgt in left
                                      if sent_idx_loc[tgt] != sent_idx_loc[head]]
                        right_inter = [tgt for tgt in right
                                       if sent_idx_loc[tgt] != sent_idx_loc[head]]
                        sides = ([right_inter, left_inter]
                                 if strategy in ['closest-intra-lr-inter-rl',
                                                 'closest-intra-rl-inter-rl']
                                 else [left_inter, right_inter])
                        # same code as 'closest-*' above
                        while any(sides):
                            if left_inter and right_inter:
                                dist_left = abs(left_inter[-1] - head)
                                dist_right = abs(right_inter[-1] - head)
                                if dist_left == dist_right:
                                    side = sides[0]
                                elif dist_left < dist_right:
                                    side = left_inter
                                else:
                                    side = right_inter
                            else:  # one side is empty
                                side = (left_inter if left_inter
                                        else right_inter)
                            # same code as lrlrlr/rlrlrl above ; make into
                            # helper function?
                            dep_cur = side.pop()
                            ranks[dep_cur] = rank_idx
                            # weakly-ordered: consecutive nuclei with
                            # same label are assumed to be part of a
                            # multinuclear relation => same rank
                            lbl_cur = dtree.labels[dep_cur]
                            nuc_cur = dtree.nucs[dep_cur]
                            if self.order == 'weak' and nuc_cur == NUC_N:
                                while (side
                                       and dtree.labels[side[-1]] == lbl_cur
                                       and dtree.nucs[side[-1]] == nuc_cur):
                                    # give same rank
                                    dep_cur = side.pop()
                                    ranks[dep_cur] = rank_idx
                            # increment rank
                            rank_idx += 1

                    elif strategy in ['sdist-edist-lr', 'sdist-edist-rl']:
                        # used 2016-09-13: sdist-edist-rl
                        # distance in sentences, then in EDUs, then pick
                        # side to break ties
                        sides = ([left, right] if strategy == 'sdist-edist-lr'
                                 else [right, left])
                        while any(sides):
                            if left and right:
                                # distances: in sentences, EDUs
                                # * next candidate on the left
                                sdist_left = abs(sent_idx_loc[left[-1]]
                                                 - sent_idx_loc[head])
                                edist_left = abs(left[-1] - head)
                                dist_left = (sdist_left, edist_left)
                                # * next candidate on the right
                                sdist_right = abs(sent_idx_loc[right[-1]]
                                                  - sent_idx_loc[head])
                                edist_right = abs(right[-1] - head)
                                dist_right = (sdist_right, edist_right)
                                # * compare
                                if dist_left == dist_right:
                                    side = sides[0]
                                elif dist_left < dist_right:
                                    side = left
                                else:
                                    side = right
                            else:  # one side is empty
                                side = left if left else right
                            # same code as lrlrlr/rlrlrl above ; make into
                            # helper function?
                            dep_cur = side.pop()
                            ranks[dep_cur] = rank_idx
                            # weakly-ordered: consecutive nuclei with
                            # same label are assumed to be part of a
                            # multinuclear relation => same rank
                            lbl_cur = dtree.labels[dep_cur]
                            nuc_cur = dtree.nucs[dep_cur]
                            if self.order == 'weak' and nuc_cur == NUC_N:
                                while (side
                                       and dtree.labels[side[-1]] == lbl_cur
                                       and dtree.nucs[side[-1]] == nuc_cur):
                                    # give same rank
                                    dep_cur = side.pop()
                                    ranks[dep_cur] = rank_idx
                            # increment rank
                            rank_idx += 1

                    else:
                        raise RstDtException('Unknown transformation strategy'
                                             ' {stg}'.format(stg=strategy))

            dt_ranks.append(ranks)
        return dt_ranks


def deptree_to_simple_rst_tree(dtree, allow_forest=False):
    r"""
    Given a dependency tree with attachment ranking and nuclearity,
    return a 'SimpleRSTTree'.
    Note that nodes are ordered by text span, so that `e1 -R-> e2` and
    `e2 -R-> e1` would both be interpreted as `R(e1, e2)`, the
    potential difference being which is treated as the nucleus and as
    the satellite.

    The conversion algorithm can be seen a two-pass descent/conversion,
    ascent/assembly one. First, we go down the dependency tree,
    annotating each node with a trivial RST tree (ie. a leaf node).
    Then, we walk back up the dependency tree, connecting RST treelets
    together, gradually replacing nodes with increasingly larger RST
    trees.

    The first tricky part of this conversion is to keep in mind
    that the conversion process involves some rotation, namely
    that this dependency structure ::

           R
        a ---> b

    (if we gloss over nuclearity and node order details) is rotated
    into something that looks like this ::


        R +-> a                                 R
          |          (or more conventionally,  / \ )
          +-> b                               a   b


    Where things get tricky is if a node in the dependency tree
    points has multiple children ::

            r1
        src +--> tgt1
            |
            |r2
            +--> tgt2
            |
            ..
            |
            |rN
            +--> tgtN


    In that case, having pivoted the `src(r1:tgt1)` link into something
    like `r1(src, tgt)`, it may not be so clear how to deal with
    subsequents targets `tgt2..tgtN`. We want to both reflect that `src`
    has a dependency link with all of these targets at the same time
    even though it's stuck as an RST child node. The trick is to see the
    process as more of a fold than a map. Keeping in mind that `src`,
    `tgt1..tgtN` are all RST treelets (`tgt1..tgtN` having been built up
    from recursive descent), the idea is to gradually replace the src
    node with an increasingly nested RST tree as we go along ::


        +----------+
        |   r1     |    r2
        |  /  \    + -+---> tgt2
        | src tgt1 |  |
        +----------+  | r3
                      +---> tgt3
                      |
                      ..
                      | rN
                      +---> tgtN



        +------------+
        |     r2     |
        |    /  \    |
        |   r1  tgt2 |    r3
        |  /  \      + -+---> tgt3
        | src tgt1   |  |
        +------------+  ..
                        | rN
                        +---> tgtN


    (and so on, until all we have left is a single RST tree).

    TODO
    ----
    * [ ] fix the signature of this function: change name or arguments
          or return type, because the current implementation returns
          either a SimpleRSTTree if allow_forest=False, or a list of
          SimpleRSTTree if allow_forest=True. This is a likely source of
          errors because SimpleRSTTrees are list-like, ie. tree[i]
          returns the i-th child of a tree node...
    """
    def walk(ancestor, subtree):
        """
        The basic descent/ascent driver of our conversion algorithm.
        Note that we are looking at three layers of the dependency
        tree at the same time.


                     r0       r1
            ancestor --> src +--> tgt1
                             |
                             |r2
                             +--> tgt2
                             |
                             ..
                             |
                             |rN
                             +--> tgtN

        The base case is if src is a leaf node (no children),
        whereupon we return a tiny tree connecting the two.

        If we do have children, we have to first obtain the
        full RST tree for src (through the folding process
        described in the docstring for the main function)
        before connecting it to its ancestor.

        Parameters
        ----------
        ancestor : SimpleRSTTree
            SimpleRSTTree of the ancestor

        subtree : int
            Index of the head of the subtree

        Returns
        -------
        res : SimpleRSTTree
            SimpleRSTTree covering ancestor and subtree.
        """
        # create tree leaf for src
        edu_src = dtree.edus[subtree]
        src = SimpleRSTTree(
            Node("leaf", (edu_src.num, edu_src.num), edu_src.text_span(),
                 "leaf"),
            [edu_src])

        # descend into each child, but note that we are folding
        # rather than mapping, ie. we threading along a nested
        # RST tree as go from sibling to sibling
        ranked_targets = dtree.deps(subtree)
        for tgt in ranked_targets:
            src = walk(src, tgt)
        if not ancestor:
            # first call: ancestor is None, subtree is the index of the
            # (presumably unique) real root
            return src

        # connect ancestor with src
        n_anc = treenode(ancestor)
        n_src = treenode(src)
        rel = dtree.labels[subtree]
        nuc = dtree.nucs[subtree]
        #
        if n_anc.span.overlaps(n_src.span):
            raise RstDtException("Span %s overlaps with %s " %
                                 (n_anc.span, n_src.span))
        else:
            if n_anc.span <= n_src.span:
                left = ancestor
                right = src
                nuc_kids = [NUC_N, nuc]
            else:
                left = src
                right = ancestor
                nuc_kids = [nuc, NUC_N]
            # nuc in SimpleRSTTree is the concatenation of the initial
            # letter of each kid's nuclearity for the relation,
            # eg. {NS, SN, NN}
            nuc = ''.join(x[0] for x in nuc_kids)
        # compute EDU span of the parent node from the kids'
        l_edu_span = treenode(left).edu_span
        r_edu_span = treenode(right).edu_span
        edu_span = (min(l_edu_span[0], r_edu_span[0]),
                    max(l_edu_span[1], r_edu_span[1]))
        txt_span = n_anc.span.merge(n_src.span)
        res = SimpleRSTTree(
            Node(nuc, edu_span, txt_span, rel),
            [left, right])
        return res

    roots = dtree.real_roots_idx()
    if not allow_forest and len(roots) > 1:
        msg = ('Cannot convert RstDepTree to SimpleRSTTree, '
               'multiple roots: {}\t{}'.format(roots, dtree.__dict__))
        raise RstDtException(msg)

    srtrees = []
    for real_root in roots:
        srtree = walk(None, real_root)
        srtrees.append(srtree)

    # for the most common case, return the tree
    if not allow_forest:
        return srtrees[0]
    # otherwise return a forest of SimpleRSTTrees ; needed for e.g.
    # intra-sentential parsing with leaky sentences, or sentence-only
    # document parsing.
    return srtrees


def deptree_to_rst_tree(dtree):
    """Create an RSTTree from an RstDepTree.

    Parameters
    ----------
    dtree: RstDepTree
        RST dependency tree, i.e. an ordered dtree.

    Returns
    -------
    ctree: RSTTree
        RST constituency tree that corresponds to the dtree.
    """
    heads = dtree.heads
    ranks = dtree.ranks
    origin = dtree.origin

    # gov -> (rank -> [deps])
    ranked_deps = defaultdict(lambda: defaultdict(list))
    for dep, (gov, rnk) in enumerate(zip(heads[1:], ranks[1:]), start=1):
        ranked_deps[gov][rnk].append(dep)

    # store pointers to substructures as they are built
    subtrees = [None for x in dtree.edus]

    # compute height of each governor in the dtree
    heights = [0 for x in dtree.edus]
    while True:
        old_heights = tuple(heights)
        for i, hd in enumerate(dtree.heads[1:], start=1):
            heights[hd] = max(heights[hd], heights[i] + 1)
        if tuple(heights) == old_heights:
            # fixpoint reached
            break
    # group nodes by their height in the dtree
    govs_by_height = defaultdict(list)
    for i, height in enumerate(heights):
        govs_by_height[height].append(i)

    # bottom-up traversal of the dtree: create sub-ctrees
    # * create leaves of the RST ctree: initialize them with the
    # label and nuclearity from the dtree
    for i in range(1, len(dtree.edus)):
        node = Node(dtree.nucs[i], (i, i), dtree.edus[i].span,
                    dtree.labels[i], context=None)  # TODO context?
        children = [dtree.edus[i]]  # WIP
        subtrees[i] = RSTTree(node, children, origin=origin)

    # * create internal nodes: for each governor, create one projection
    # per rank of dependents ; each time a projection node is created,
    # we use the set of dependencies to overwrite the nuc and label of
    # its children
    for height in range(1, max(heights)):  # leave fake root out, see below
        nodes = govs_by_height[height]
        for gov in nodes:
            # max_rnk = max(ranked_deps[gov].keys())
            for rnk, deps in sorted(ranked_deps[gov].items()):
                # overwrite the nuc and lbl of the head node, using the
                # dependencies of this rank
                dep_nucs = [dtree.nucs[x] for x in deps]
                dep_lbls = [dtree.labels[x] for x in deps]
                if all(x == NUC_N for x in dep_nucs):
                    # all nuclei must have the same label, to denote
                    # a unique multinuclear relation
                    assert len(set(dep_lbls)) == 1
                    gov_lbl = dep_lbls[0]
                elif all(x == NUC_S for x in dep_nucs):
                    gov_lbl = 'span'
                else:
                    raise ValueError('Deps have different nuclearities')
                gov_node = subtrees[gov].label()
                gov_node.nuclearity = NUC_N
                gov_node.rel = gov_lbl
                # create one projection node for the head + the dependencies
                # of this rank
                proj_lbl = dtree.labels[gov]
                proj_nuc = dtree.nucs[gov]
                proj_children = [subtrees[x] for x in sorted([gov] + deps)]
                proj_edu_span = (proj_children[0].label().edu_span[0],
                                 proj_children[-1].label().edu_span[1])
                proj_txt_span = Span(proj_children[0].label().span.char_start,
                                     proj_children[-1].label().span.char_end)
                proj_node = Node(proj_nuc, proj_edu_span, proj_txt_span,
                                 proj_lbl, context=None)  # TODO context?
                subtrees[gov] = RSTTree(proj_node, proj_children,
                                        origin=origin)
    # create top node and whole tree
    # this is where we handle the fake root
    gov = 0
    proj_lbl = '---'  # 2016-12-02: switch from "ROOT" to "---" so that
    # _pred and _true have the same labels for their root nodes
    proj_nuc = NUC_R
    if (ranked_deps[gov].keys() == [1]
        and len(ranked_deps[gov][1]) == 1):
        # unique real root => use its projection as the root of the ctree
        unique_real_root = ranked_deps[gov][1][0]
        # proj = subtrees[unique_real_root].label()
        proj_node.nuclearity = proj_nuc
        proj_node.rel = proj_lbl
        subtrees[0] = subtrees[unique_real_root]
    else:
        # > 1 real root: create projections until we span all
        # 2016-09-14 disable support for >1 real root
        raise ValueError("Fragile: RSTTree from dtree with >1 real root")
        #
        # max_rnk = max(ranked_deps[gov].keys())
        for rnk, deps in sorted(ranked_deps[gov].items()):
            # overwrite the nuc and lbl of the head node, using the
            # dependencies of this rank
            dep_nucs = [dtree.nucs[x] for x in deps]
            dep_lbls = [dtree.labels[x] for x in deps]
            if all(x == NUC_N for x in dep_nucs):
                # all nuclei must have the same label, to denote
                # a unique multinuclear relation
                assert len(set(dep_lbls)) == 1
                gov_lbl = dep_lbls[0]
            elif all(x == NUC_S for x in dep_nucs):
                gov_lbl = 'span'
            else:
                raise ValueError('Deps have different nuclearities')
            gov_node = subtrees[gov].label()
            gov_node.nuclearity = NUC_N
            gov_node.rel = gov_lbl
            # create one projection node for the head + the dependencies
            # of this rank
            proj_lbl = dtree.labels[gov]
            proj_nuc = dtree.nucs[gov]
            proj_children = [subtrees[x] for x in sorted([gov] + deps)]
            proj_edu_span = (proj_children[0].label().edu_span[0],
                             proj_children[-1].label().edu_span[1])
            proj_txt_span = Span(proj_children[0].label().span.char_start,
                                 proj_children[-1].label().span.char_end)
            proj_node = Node(proj_nuc, proj_edu_span, proj_txt_span,
                             proj_lbl, context=None)  # TODO context?
            subtrees[gov] = RSTTree(proj_node, proj_children,
                                    origin=origin)
    # final RST ctree
    rst_tree = subtrees[0]
    return rst_tree
