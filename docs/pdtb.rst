
Educe
=====

`Educe <http://kowey.github.io/educe>`__ is a library for working with a
variety of discourse corpora. This tutorial aims to show what using
educe would be like when working with the `Penn Discourse
Treebank <http://www.seas.upenn.edu/~pdtb/>`__ corpus.

Installation
------------

.. code:: shell

    git clone https://github.com/kowey/educe.git
    cd educe
    pip install -r requirements.txt

Note: these instructions assume you are running within a `virtual
environment <http://virtualenv.readthedocs.org/en/latest/>`__. If not,
and if you have permission denied errors, replace ``pip`` with
``sudo pip``.

Tutorial setup
--------------

This tutorial require that you have a local copy of the PDTB. For
purposes of this tutorial, you will need to link this into the data
directory, for example

::

    ln -s $HOME/CORPORA/pdtb_v2 data

Optionnally, to match the pdtb text spans to their analysis in the Penn
Treebank, you need to have a local copy of the PTB at the same location

::

    ln -s $HOME/CORPORA/PTBIII data

Tutorial in browser (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial can either be followed along with the command line and
your favourite text editor, or embedded in an interactive webpage via
iPython:

.. code:: shell

    pip install ipython
    cd tutorials
    ipython notebook

.. code:: python

    # some helper functions for the tutorial below
    
    def show_type(rel):
        "short string for a relation type"
        return type(rel).__name__[:-8]  # remove "Relation"
    
    def highlight(astring, color=1):
        "coloured text"
        return("\x1b[3{color}m{str}\x1b[0m".format(color=color, str=astring))

Reading corpus files (PDTB)
---------------------------

NB: unfortunately, at the time of this writing, PDTB support in educe is
very much behind and rather inconsistent with that of the other corpora.
Apologies for the mess!

.. code:: python

    from __future__ import print_function
    import educe.pdtb
    
    # relative to the educe docs directory
    data_dir = '../data'
    corpus_dir = '{dd}/pdtb_v2/data'.format(dd=data_dir)
    
    # read a small sample of the pdtb
    reader = educe.pdtb.Reader(corpus_dir)
    anno_files = reader.filter(reader.files(),
                               lambda k: k.doc.startswith('wsj_231'))
    corpus = reader.slurp(anno_files, verbose=True)
        
    # print the first five rel types we read from each doc
    for key in corpus.keys()[:10]:
        doc = corpus[key]
        rtypes = [show_type(r) for r in doc]
        print("[{0}] {1}".format(key.doc, " ".join(rtypes[:5])))


.. parsed-literal::

    Slurping corpus dir [7/8]

.. parsed-literal::

    [wsj_2315] Explicit Implicit Entity Explicit Implicit
    [wsj_2311] Implicit
    [wsj_2316] Explicit Implicit Implicit Implicit Explicit
    [wsj_2310] Entity
    [wsj_2319] Explicit
    [wsj_2317] Implicit Implicit Explicit Implicit Explicit
    [wsj_2313] Entity Explicit Explicit Implicit Explicit
    [wsj_2314] Explicit Explicit Implicit Explicit Entity


.. parsed-literal::

    Slurping corpus dir [8/8 done]


What's a corpus?
----------------

A corpus is a dictionary from ``FileId`` keys to representation of PDTB
documents.

Keys
~~~~

A key has several fields meant to distinguish different annotated
documents from each other. In the case of the PDTB, the only field of
interest is ``doc``, a Wall Street journal article number as you might
find in the PTB.

.. code:: python

    ex_key = educe.pdtb.mk_key('wsj_2314')
    ex_doc = corpus[ex_key]
    
    print(ex_key)
    print(ex_key.__dict__)


.. parsed-literal::

    wsj_2314 [None] discourse unknown
    {'doc': 'wsj_2314', 'subdoc': None, 'annotator': 'unknown', 'stage': 'discourse'}


Documents
~~~~~~~~~

At some point in the future, the representation of a document may change
to something a bit higher level and easier to work with. For now, a
“document” in the educe PDTB sense consists of a list of relations, each
relation having a low-level representation that hews fairly closely to
the grammar described in the PDTB annotation manual.

**TIP**: At least until educe grows a more educe-like uniform
representation of PDTB annotations, a very useful resource to look at
when working with the PDTB may be The Penn Discourse Treebank 2.0
Annotation Manual, sections 6.3.1 to 6.3.5 (Description of PDTB
representation format → File format → General outline…).

.. code:: python

    lr = [r for r in ex_doc]
    r0 = lr[0]
    type(r0).__name__




.. parsed-literal::

    'ExplicitRelation'



Relations
~~~~~~~~~

There are five types of relation annotation: explicit, implicit, altlex,
entity, no (as in no relation). These are described in further detail in
the PDTB annotation manual. Here's well try to sketch out some of the
important properties.

The main thing to notice is that the 5 types of annotation not have very
much in common with each other, but they have many overlapping pieces
(see table in the `educe.pdtb
docs <https://educe.readthedocs.org/en/latest/api-doc/educe.pdtb.html>`__)

-  a relation instance always has two arguments (these can be selected
   as ``arg1`` and ``arg2``)

.. code:: python

    def display_rel(r):
        "pretty print a relation instance"
    
        rtype = show_type(r)
        
        if rtype == "Explicit":
            conn = highlight(r.connhead)
        elif rtype == "Implicit":
            conn = "{rtype} {conn1}".format(rtype=rtype,
                                            conn1=highlight(str(r.connective1)))
        elif rtype == "AltLex":
            conn = "{rtype} {sem1}".format(rtype=rtype,
                                           sem1=highlight(r.semclass1))
        else:
            conn = rtype
    
        fmt = "{src}\n \t ---[{label}]---->\n \t\t\t{tgt}"
        return(fmt.format(src=highlight(r.arg1.text, 2),
                          label=conn,
                          tgt=highlight(r.arg2.text, 2)))
            
    
    print(display_rel(r0))


.. parsed-literal::

    [32mQuantum Chemical Corp. went along for the ride[0m
     	 ---[[31mConnective(when | Temporal.Synchrony)[0m]---->
     			[32mthe price of plastics took off in 1987[0m


.. code:: python

    r0.connhead.text




.. parsed-literal::

    u'when'



Gorn addresses
--------------

.. code:: python

    # print the first seven gorn addresses for the first argument of the first
    # 5 rels we read from each doc
    for key in corpus.keys()[:3]:
        doc = corpus[key]
        rels = doc[:5]
        print(key.doc)
        for r in doc[:5]:
            print("\t{0}".format(r.arg1.gorn[:7]))


.. parsed-literal::

    wsj_2315
    	[0.0, 0.1.0, 0.1.1.0, 0.1.1.1, 0.1.1.2, 0.2]
    	[1.1.1]
    	[3]
    	[5.1.1.1.0]
    	[6.0, 6.1.0, 6.1.1.0, 6.1.1.1.0, 6.1.1.1.1, 6.1.1.1.2, 6.1.1.1.3.0]
    wsj_2311
    	[0]
    wsj_2316
    	[0.0.0, 0.0.1, 0.0.3, 0.1, 0.2]
    	[2.0.0, 2.0.1, 2.0.3, 2.1, 2.2]
    	[4]
    	[5.3.4.1.1.2.2.2]
    	[5.3.4]


Penn Treebank integration
-------------------------

.. code:: python

    from educe.pdtb import ptb
    
    # confusingly, this is not an educe corpus reader, but the NLTK
    # bracketed reader.  Sorry
    ptb_reader = ptb.reader('{dd}/PTBIII/parsed/mrg/wsj/'.format(dd=data_dir))
    ptb_trees = {}
    for key in corpus.keys()[:3]:
        ptb_trees[key] = ptb.parse_trees(corpus, key, ptb_reader)
        print("{0}...".format(str(ptb_trees[key])[:100]))


.. parsed-literal::

    [Tree('S', [Tree('NP-SBJ-1', [Tree('NNP', ['RJR']), Tree('NNP', ['Nabisco']), Tree('NNP', ['Inc.'])]...
    [Tree('S', [Tree('NP-SBJ', [Tree('NNP', ['CONCORDE']), Tree('JJ', ['trans-Atlantic']), Tree('NNS', [...
    [Tree('S', [Tree('NP-SBJ', [Tree('NP', [Tree('DT', ['The']), Tree('NNP', ['U.S.'])]), Tree(',', [','...


.. code:: python

    !ls ../data/PTBIII/parsed/mrg/wsj/


.. parsed-literal::

    [34m00[m[m [34m01[m[m [34m02[m[m [34m03[m[m [34m04[m[m [34m05[m[m [34m06[m[m [34m07[m[m [34m08[m[m [34m09[m[m [34m10[m[m [34m11[m[m [34m12[m[m [34m13[m[m [34m14[m[m [34m15[m[m [34m16[m[m [34m17[m[m [34m18[m[m [34m19[m[m [34m20[m[m [34m21[m[m [34m22[m[m [34m23[m[m [34m24[m[m


.. code:: python

    def pick_subtree(tree, gparts):
        if gparts:
            return pick_subtree(tree[gparts[0]], gparts[1:])
        else:
            return tree
    
    # print the first seven gorn addresses for the first argument of the first
    # 5 rels we read from each doc, along with the corresponding subtree
    ndocs = 1
    nrels = 3
    ngorn = -1
    
    for key in corpus.keys()[:1]:
        doc = corpus[key]
        rels = doc[:nrels]
        ptb_tree = ptb_trees[key]
        print("======="+key.doc)
        for i,r in enumerate(doc[:nrels]):
            print("---- relation {0}".format(i+1))
            print(display_rel(r))
            
            for (i,arg) in enumerate([r.arg1,r.arg2]):
                print(".... arg {0}".format(i+1))
                glist = arg.gorn # arg.gorn[:ngorn]
                subtrees = [pick_subtree(ptb_tree, g.parts) for g in glist]
                for gorn, subtree in zip(glist, subtrees):
                    print("{0}\n{1}".format(gorn, str(subtree)))


.. parsed-literal::

    =======wsj_2315
    ---- relation 1
    [32mRJR Nabisco Inc. is disbanding its division responsible for buying network advertising time[0m
     	 ---[[31mConnective(after | Temporal.Asynchronous.Succession)[0m]---->
     			[32mmoving 11 of the group's 14 employees to New York from Atlanta[0m
    .... arg 1
    0.0
    (NP-SBJ-1 (NNP RJR) (NNP Nabisco) (NNP Inc.))
    0.1.0
    (VBZ is)
    0.1.1.0
    (VBG disbanding)
    0.1.1.1
    (NP
      (NP (PRP$ its) (NN division))
      (ADJP
        (JJ responsible)
        (PP
          (IN for)
          (S-NOM
            (NP-SBJ (-NONE- *))
            (VP
              (VBG buying)
              (NP (NN network) (NN advertising) (NN time)))))))
    0.1.1.2
    (, ,)
    0.2
    (. .)
    .... arg 2
    0.1.1.3.2
    (S-NOM
      (NP-SBJ (-NONE- *-1))
      (VP
        (VBG moving)
        (NP
          (NP (CD 11))
          (PP
            (IN of)
            (NP
              (NP (DT the) (NN group) (POS 's))
              (CD 14)
              (NNS employees))))
        (PP-DIR (TO to) (NP (NNP New) (NNP York)))
        (PP-DIR (IN from) (NP (NNP Atlanta)))))
    ---- relation 2
    [32mthat it is shutting down the RJR Nabisco Broadcast unit, and dismissing its 14 employees, in a move to save money[0m
     	 ---[Implicit [31mConnective(in addition | Expansion.Conjunction)[0m]---->
     			[32mRJR is discussing its network-buying plans with its two main advertising firms, FCB/Leber Katz and McCann Erickson[0m
    .... arg 1
    1.1.1
    (SBAR
      (IN that)
      (S
        (NP-SBJ (PRP it))
        (VP
          (VBZ is)
          (VP
            (VP
              (VBG shutting)
              (PRT (RP down))
              (NP
                (DT the)
                (NNP RJR)
                (NNP Nabisco)
                (NNP Broadcast)
                (NN unit)))
            (, ,)
            (CC and)
            (VP (VBG dismissing) (NP (PRP$ its) (CD 14) (NNS employees)))
            (, ,)
            (PP-LOC
              (IN in)
              (NP
                (DT a)
                (NN move)
                (S
                  (NP-SBJ (-NONE- *))
                  (VP (TO to) (VP (VB save) (NP (NN money)))))))))))
    .... arg 2
    2.1.1
    (SBAR
      (-NONE- 0)
      (S
        (NP-SBJ (NNP RJR))
        (VP
          (VBZ is)
          (VP
            (VBG discussing)
            (NP (PRP$ its) (JJ network-buying) (NNS plans))
            (PP
              (IN with)
              (NP
                (NP
                  (PRP$ its)
                  (CD two)
                  (JJ main)
                  (NN advertising)
                  (NNS firms))
                (, ,)
                (NP
                  (NP (NNP FCB\/Leber) (NNP Katz))
                  (CC and)
                  (NP (NNP McCann) (NNP Erickson)))))))))
    ---- relation 3
    [32mWe found with the size of our media purchases that an ad agency could do just as good a job at significantly lower cost," said the spokesman, who declined to specify how much RJR spends on network television time[0m
     	 ---[Entity]---->
     			[32mAn executive close to the company said RJR is spending about $140 million on network television time this year, down from roughly $200 million last year[0m
    .... arg 1
    3
    (SINV
      (`` ``)
      (S-TPC-3
        (NP-SBJ (PRP We))
        (VP
          (VBD found)
          (PP
            (IN with)
            (NP
              (NP (DT the) (NN size))
              (PP (IN of) (NP (PRP$ our) (NNS media) (NNS purchases)))))
          (SBAR
            (IN that)
            (S
              (NP-SBJ (DT an) (NN ad) (NN agency))
              (VP
                (MD could)
                (VP
                  (VB do)
                  (NP (ADJP (RB just) (RB as) (JJ good)) (DT a) (NN job))
                  (PP
                    (IN at)
                    (NP (ADJP (RB significantly) (JJR lower)) (NN cost)))))))))
      (, ,)
      ('' '')
      (VP (VBD said) (S (-NONE- *T*-3)))
      (NP-SBJ
        (NP (DT the) (NN spokesman))
        (, ,)
        (SBAR
          (WHNP-1 (WP who))
          (S
            (NP-SBJ-4 (-NONE- *T*-1))
            (VP
              (VBD declined)
              (S
                (NP-SBJ (-NONE- *-4))
                (VP
                  (TO to)
                  (VP
                    (VB specify)
                    (SBAR
                      (WHNP-2 (WRB how) (JJ much))
                      (S
                        (NP-SBJ (NNP RJR))
                        (VP
                          (VBZ spends)
                          (NP (-NONE- *T*-2))
                          (PP-CLR
                            (IN on)
                            (NP (NN network) (NN television) (NN time)))))))))))))
      (. .))
    .... arg 2
    4
    (S
      (NP-SBJ
        (NP (DT An) (NN executive))
        (ADJP (RB close) (PP (TO to) (NP (DT the) (NN company)))))
      (VP
        (VBD said)
        (SBAR
          (-NONE- 0)
          (S
            (NP-SBJ (NNP RJR))
            (VP
              (VBZ is)
              (VP
                (VBG spending)
                (NP
                  (NP
                    (QP (RB about) ($ $) (CD 140) (CD million))
                    (-NONE- *U*))
                  (ADVP (-NONE- *ICH*-1)))
                (PP-CLR
                  (IN on)
                  (NP (NN network) (NN television) (NN time)))
                (NP-TMP (DT this) (NN year))
                (, ,)
                (ADVP-1
                  (RB down)
                  (PP
                    (IN from)
                    (NP
                      (NP
                        (QP (RB roughly) ($ $) (CD 200) (CD million))
                        (-NONE- *U*))
                      (NP-TMP (JJ last) (NN year))))))))))
      (. .))


.. code:: python

    print(subtree.flatten())
    print(subtree.leaves())


.. parsed-literal::

    (S
      An
      executive
      close
      to
      the
      company
      said
      0
      RJR
      is
      spending
      about
      $
      140
      million
      *U*
      *ICH*-1
      on
      network
      television
      time
      this
      year
      ,
      down
      from
      roughly
      $
      200
      million
      *U*
      last
      year
      .)
    [u'An', u'executive', u'close', u'to', u'the', u'company', u'said', u'0', u'RJR', u'is', u'spending', u'about', u'$', u'140', u'million', u'*U*', u'*ICH*-1', u'on', u'network', u'television', u'time', u'this', u'year', u',', u'down', u'from', u'roughly', u'$', u'200', u'million', u'*U*', u'last', u'year', u'.']


.. code:: python

    from copy import copy
    t = copy(subtree)
    print("constituent = "+ highlight(t.label()))
    for i in range(len(subtree)):
        print(i)
        print(t.pop())


.. parsed-literal::

    constituent = [31mS[0m
    0
    (. .)
    1
    (VP
      (VBD said)
      (SBAR
        (-NONE- 0)
        (S
          (NP-SBJ (NNP RJR))
          (VP
            (VBZ is)
            (VP
              (VBG spending)
              (NP
                (NP
                  (QP (RB about) ($ $) (CD 140) (CD million))
                  (-NONE- *U*))
                (ADVP (-NONE- *ICH*-1)))
              (PP-CLR
                (IN on)
                (NP (NN network) (NN television) (NN time)))
              (NP-TMP (DT this) (NN year))
              (, ,)
              (ADVP-1
                (RB down)
                (PP
                  (IN from)
                  (NP
                    (NP
                      (QP (RB roughly) ($ $) (CD 200) (CD million))
                      (-NONE- *U*))
                    (NP-TMP (JJ last) (NN year))))))))))
    2
    (NP-SBJ
      (NP (DT An) (NN executive))
      (ADJP (RB close) (PP (TO to) (NP (DT the) (NN company)))))


.. code:: python

    from copy import copy
    t = copy(subtree)
    
    def expand(subtree):
        if type(subtree) is unicode:
            print(subtree)
        else:
            print("constituent = "+ highlight(subtree.label()))
            for i, st in enumerate(subtree):
                #print(i)
                expand(st)
      
    expand(t)


.. parsed-literal::

    constituent = [31mS[0m
    constituent = [31mNP-SBJ[0m
    constituent = [31mNP[0m
    constituent = [31mDT[0m
    An
    constituent = [31mNN[0m
    executive
    constituent = [31mADJP[0m
    constituent = [31mRB[0m
    close
    constituent = [31mPP[0m
    constituent = [31mTO[0m
    to
    constituent = [31mNP[0m
    constituent = [31mDT[0m
    the
    constituent = [31mNN[0m
    company
    constituent = [31mVP[0m
    constituent = [31mVBD[0m
    said
    constituent = [31mSBAR[0m
    constituent = [31m-NONE-[0m
    0
    constituent = [31mS[0m
    constituent = [31mNP-SBJ[0m
    constituent = [31mNNP[0m
    RJR
    constituent = [31mVP[0m
    constituent = [31mVBZ[0m
    is
    constituent = [31mVP[0m
    constituent = [31mVBG[0m
    spending
    constituent = [31mNP[0m
    constituent = [31mNP[0m
    constituent = [31mQP[0m
    constituent = [31mRB[0m
    about
    constituent = [31m$[0m
    $
    constituent = [31mCD[0m
    140
    constituent = [31mCD[0m
    million
    constituent = [31m-NONE-[0m
    *U*
    constituent = [31mADVP[0m
    constituent = [31m-NONE-[0m
    *ICH*-1
    constituent = [31mPP-CLR[0m
    constituent = [31mIN[0m
    on
    constituent = [31mNP[0m
    constituent = [31mNN[0m
    network
    constituent = [31mNN[0m
    television
    constituent = [31mNN[0m
    time
    constituent = [31mNP-TMP[0m
    constituent = [31mDT[0m
    this
    constituent = [31mNN[0m
    year
    constituent = [31m,[0m
    ,
    constituent = [31mADVP-1[0m
    constituent = [31mRB[0m
    down
    constituent = [31mPP[0m
    constituent = [31mIN[0m
    from
    constituent = [31mNP[0m
    constituent = [31mNP[0m
    constituent = [31mQP[0m
    constituent = [31mRB[0m
    roughly
    constituent = [31m$[0m
    $
    constituent = [31mCD[0m
    200
    constituent = [31mCD[0m
    million
    constituent = [31m-NONE-[0m
    *U*
    constituent = [31mNP-TMP[0m
    constituent = [31mJJ[0m
    last
    constituent = [31mNN[0m
    year
    constituent = [31m.[0m
    .


Work in progress
----------------

This tutorial is very much a work in progress. Moreover, support for the
PDTB in educe is still very incomplete. So it's very much a moving
target.
