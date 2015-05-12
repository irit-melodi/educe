
Educe
=====

`Educe <http://kowey.github.io/educe>`__ is a library for working with a
variety of discourse corpora. This tutorial aims to show what using
educe would be like when working with the
`STAC <http://www.irit.fr/STAC/>`__ corpus.

We'll be working with a tiny fragment of the corpus included with educe.
You may find it useful to symlink your larger copy from the STAC
distribution and modify this tutorial accordingly.

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

Tutorial in browser (optional)
------------------------------

This tutorial can either be followed along with the command line and
your favourite text editor, or embedded in an interactive webpage via
iPython:

.. code:: shell

    pip install ipython
    cd tutorials
    ipython notebook

.. code:: python

    # some helper functions for the tutorial below
    
    def text_snippet(text):
        "short text fragment"
        if len(text) < 43:
            return text
        else:
            return "{0}...{1}".format(text[:20], text[-20:])
    
    def highlight(astring, color=1):
        "coloured text"
        return("\x1b[3{color}m{str}\x1b[0m".format(color=color, str=astring))

Reading corpus files (STAC)
---------------------------

Typically, the first thing we want to do when working in educe is to
read the corpus in. This can be a bit slow, but as we will see later on,
we can speed things up if we know what we're looking for.

.. code:: python

    from __future__ import print_function
    import educe.stac
    
    # relative to the educe docs directory
    data_dir = '../data'
    corpus_dir = '{dd}/stac-sample'.format(dd=data_dir)
    
    # read everything from our sample
    reader = educe.stac.Reader(corpus_dir)
    corpus = reader.slurp(verbose=True)
    
    # print a text fragment from the first ten files we read
    for key in corpus.keys()[:10]:
        doc = corpus[key]
        print("[{0}] {1}".format(key, doc.text()[:50]))


.. parsed-literal::

    Slurping corpus dir [99/100]

.. parsed-literal::

    [s1-league2-game1 [05] unannotated None]  199 : sabercat : anyone any clay? 200 : IG : nope
    [s1-league2-game1 [13] units hjoseph]  521 : sabercat : skinnylinny 522 : sabercat : som
    [s1-league2-game1 [10] units hjoseph]  393 : skinnylinny : Shall we extend? 394 : saberc
    [s1-league2-game1 [11] discourse hjoseph]  450 : skinnylinny : Argh 451 : skinnylinny : How 
    [s1-league2-game1 [10] unannotated None]  393 : skinnylinny : Shall we extend? 394 : saberc
    [s1-league2-game1 [02] units lpetersen]  75 : sabercat : anyone has any wood? 76 : skinnyl
    [s1-league2-game1 [14] units SILVER]  577 : sabercat : skinny 578 : sabercat : I need 2
    [s1-league2-game3 [03] discourse lpetersen]  151 : amycharl : got wood anyone? 152 : sabercat 
    [s1-league2-game1 [10] discourse hjoseph]  393 : skinnylinny : Shall we extend? 394 : saberc
    [s1-league2-game1 [12] units SILVER]  496 : sabercat : yes! 497 : sabercat : :D 498 : s


.. parsed-literal::

    Slurping corpus dir [100/100 done]


Faster reading
~~~~~~~~~~~~~~

If you know that you only want to work with a subset of the corpus
files, you can pre-filter the corpus before reading the files.

It helps to know here that an educe corpus is a mapping from `file id
keys <https://educe.readthedocs.org/en/latest/api-doc/educe.html#educe.corpus.FileId>`__
to Documents. The ``FileId`` tells us what makes a Document distinct
from another:

-  document (eg. s1-league2-game1): in STAC, the game that was played
   (here, season 1, league 2, game 1)
-  subdocument (eg. 05): a mostly arbitrary subdivision of the documents
   motivated by technical constraints (overly large documents would
   cause our annotation tool to crash)
-  stage (eg. units, discourse, parsed): the kinds of annotations
   available in the document
-  annotator (eg. hjoseph): the main annotator for a document (gold
   standard documents have the distinguished annotators, BRONZE, SILVER,
   or GOLD)

NB: unfortunately we have overloaded the word “document” here. When
talking about file ids, “document” refers to a whole game. But when
talking about actual annotation objects an educe Document actually
corresponds to a specific combination of document, subdocument, stage,
and annotator

.. code:: python

    import re
    
    # nb: you can import this function from educe.stac.corpus
    def is_metal(fileid):  
        "is this a gold standard(ish) annotation file?"
        anno = fileid.annotator or ""
        return anno.lower() in ["bronze", "silver", "gold"]
        
    # pick out gold-standard documents
    subset = reader.filter(reader.files(), 
                           lambda k: is_metal(k) and int(k.subdoc) < 4)
    corpus_subset = reader.slurp(subset, verbose=True)
    for key in corpus_subset:
        doc = corpus_subset[key]
        print("{0}: {1}".format(key, doc.text()[:50]))


.. parsed-literal::

    Slurping corpus dir [11/12]

.. parsed-literal::

    s1-league2-game1 [01] units SILVER:  1 : sabercat : btw, are we playing without the ot
    s1-league2-game1 [01] discourse SILVER:  1 : sabercat : btw, are we playing without the ot
    s1-league2-game1 [02] discourse SILVER:  75 : sabercat : anyone has any wood? 76 : skinnyl
    s1-league2-game3 [01] discourse BRONZE:  1 : amycharl : i made it! 2 : amycharl : did the 
    s1-league2-game1 [03] discourse SILVER:  109 : sabercat : well done! 110 : IG : More clay!
    s1-league2-game3 [02] units BRONZE:  73 : sabercat : skinny, got some ore? 74 : skinny
    s1-league2-game3 [01] units BRONZE:  1 : amycharl : i made it! 2 : amycharl : did the 
    s1-league2-game1 [02] units SILVER:  75 : sabercat : anyone has any wood? 76 : skinnyl
    s1-league2-game3 [02] discourse BRONZE:  73 : sabercat : skinny, got some ore? 74 : skinny
    s1-league2-game1 [03] units SILVER:  109 : sabercat : well done! 110 : IG : More clay!
    s1-league2-game3 [03] discourse BRONZE:  151 : amycharl : got wood anyone? 152 : sabercat 
    s1-league2-game3 [03] units BRONZE:  151 : amycharl : got wood anyone? 152 : sabercat 


.. parsed-literal::

    Slurping corpus dir [12/12 done]


.. code:: python

    from educe.corpus import FileId
    
    # pick out an example document to work with creating FileIds by hand
    # is not something we would typically do (normally we would just iterate
    # through a corpus), but it's useful for illustration
    ex_key = FileId(doc='s1-league2-game3',
                    subdoc='03',
                    stage='units',
                    annotator='BRONZE')
    ex_doc = corpus[ex_key]
    print(ex_key)


.. parsed-literal::

    s1-league2-game3 [03] units BRONZE


Standing off
------------

Most annotations in the STAC corpus are `educe standoff
annotations <http://educe.readthedocs.org/en/latest/api-doc/educe.html#educe.annotation.Standoff>`__.
In educe terms, this means that they (perhaps indirectly) extend the
``educe.annotation.Standoff`` class and provide a ``text_span()``
function. Much of our reasoning around annotations essentially consists
of checking that their text spans overlap or enclose each other.

As for the text spans, these refer to the raw text saved in files with
an ``.ac`` extension (eg. ``s1-league1-game3.ac``). In the `Glozz
annotation tool <http://www.glozz.org>`__, these ``.ac`` text files form
a pair with their ``.aa`` xml counterparts. Multiple annotation files
can point to the same text file.

There are also some annotations that come from 3rd party tools, which we
will uncover later.

Documents and EDUs
------------------

A document is a sort of giant annotation that contains three other kinds
of annotation

-  units - annotations that directly cover a span of text (EDUs,
   Resources, but also turns, dialogues)
-  relations - annotations that point from one annotation to another
-  schemas - annotations that point to a set of annotations

To start things off, we'll focus on one type of unit-level annotation,
the Elementary Discourse Unit

.. code:: python

    def preview_unit(doc, anno):
        "the default str(anno) can be a bit overwhelming"
        preview = "{span: <11} {id: <20} [{type: <12}] {text}"
        text = doc.text(anno.text_span())
        return preview.format(id=anno.local_id(),
                              type=anno.type,
                              span=anno.text_span(),
                              text=text_snippet(text))
    
    print("Example units")
    print("-------------")
    seen = set()
    for anno in ex_doc.units:
        if anno.type not in seen:
            seen.add(anno.type)
            print(preview_unit(ex_doc, anno))
        
    print()
    print("First few EDUs")
    print("--------------")
    for anno in filter(educe.stac.is_edu, ex_doc.units)[:4]:
        print(preview_unit(ex_doc, anno))
    



.. parsed-literal::

    Example units
    -------------
    (1,34)      stac_1368693094      [paragraph   ] 151 : amycharl : got wood anyone?
    (52,66)     stac_1368693099      [Accept      ] yep, for what?
    (117,123)   stac_1368693105      [Refusal     ] no way
    (189,191)   stac_1368693114      [Other       ] :)
    (209,210)   stac_1368693117      [Counteroffer] ?
    (659,668)   stac_1368693162      [Offer       ] how much?
    (22,26)     asoubeille_1374939590843 [Resource    ] wood
    (35,66)     stac_1368693098      [Turn        ] 152 : sabercat : yep, for what?
    (0,266)     stac_1368693124      [Dialogue    ]  151 : amycharl : go...cat : yep, thank you
    
    First few EDUs
    --------------
    (52,66)     stac_1368693099      [Accept      ] yep, for what?
    (117,123)   stac_1368693105      [Refusal     ] no way
    (163,171)   stac_1368693111      [Accept      ] could be
    (189,191)   stac_1368693114      [Other       ] :)


--------------

TODO
====

Everything below this point should be considered to be in a
scratch/broken state. It needs to ported over from its RST/DT
considerations to STAC

To do:

-  standing off (ac/aa) - shared aa
-  layers (units/discourse)
-  working with relations and schemas
-  grabbing resources etc (example of working with unit level
   annotation)
-  synchronising layers (grabbing the dialogue act and relations at the
   same time)
-  external annotations (postags, parse trees)
-  working with hypergraphs (implementing ``_repr_png()_`` would be
   pretty sweet)

--------------

Tree searching
~~~~~~~~~~~~~~

The same span enclosure logic can be used to search parse trees for
particular constituents, verb phrases. Alternatively, you can use the
the ``topdown`` method provided by educe trees. This returns just the
largest constituent for which some predicate is true. It optionally
accepts an additional argument to cut off the search when it is clearly
out of bounds.

Conclusion
----------

In this tutorial, we've explored a couple of basic educe concepts, which
we hope will enable you to extract some data from your discourse
corpora, namely

-  reading corpus data (and pre-filtering)
-  standoff annotations
-  searching by span enclosure, overlapping
-  working with trees
-  combining annotations from different sources

The concepts above should transfer to whatever discourse corpus you are
working with (that educe supports, or that you are prepared to supply a
reader for).

Work in progress
~~~~~~~~~~~~~~~~

This tutorial is very much a work in progress (last update: 2014-09-19).
Educe is a bit of a moving target, so `let me
know <https://github.com/kowey/educe/issues>`__ if you run into any
trouble!

See also
~~~~~~~~

stac-util
^^^^^^^^^

Some of the things you may want to do with the STAC corpus may already
exist in the stac-util command line tool. stac-util is meant to be a
sort of Swiss Army Knife, providing tools for editing the corpus. The
query tools are more likely to be of interest:

-  text: display text and edu/dialogue segmentation in a friendly way
-  graph: draw discourse graphs with graphviz (arrows for relations,
   boxes for CDUs, etc)
-  filter-graph: visualise instances of relations (eg. Question answer
   pair)
-  count: generate statistics about the corpus

See ``stac-util --help`` for more details.

External tool support
^^^^^^^^^^^^^^^^^^^^^

Educe has some support for reading data from outside the discourse
corpus proper. For example, if you run the stanford corenlp parser on
the raw text, you can read them back into educe-style
``ConstituencyTree`` and ``DependencyTree`` annotations. See
`educe.external <https://educe.readthedocs.org/en/latest/api-doc/educe.external.html>`__
for details.

If you have a part of speech tagger that you would like to use, the
``educe.external.postag`` module may be useful for representing the
annotations that come out of it

You can also add support for your own tools by creating annotations that
extend ``Standoff``, directly or otherwise.

