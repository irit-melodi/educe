"""
*Note: At the time of this writing, this is a slightly
idealised representation of the package.  See below for notes on
where things get a bit messier*

The educe library provides utilities for working with annotated discourse
corpora.  It has a three-layer structure:

* base layer (files, annotations, fusion, graphs)
* tool layer (specific to tools, file formats, etc)
* project layer (specific to particular corpora, currently stac)

Layers
~~~~~~
Working our way up the tower, the base layer provides four sublayers:

* file management (educe.corpus): basic model for corpus traversal,
  for selecting slices of the corpus

* annotation: (educe.annotation), representation of annotated
  texts, adhering closely to whatever annotation tool produced it.

* fusion (in progress): connections between annotations on different layers
  (eg. on speech acts for text spans, discourse relations), or from different
  tools (eg. from a POS tagger, a parser, etc)

* graph (educe.graph): high-level/abstract representation of discourse
  structure, allowing for queries on the structures themselves (eg. give me all
  pairs for discourse units separated by at most 3 nodes in the graph)

Building on the base layer, we have modules that are specific to a particular
set of annotation tools, currently this is only `educe.glozz`.  We aim to add
modules sparingly.

Finally, on top of this, we have the project layer (eg. `educe.stac`) which
keeps track of conventions specific to this particular corpus.  The hope
would be for most of your script writing to deal with this layer directly,
eg. for STAC ::

               stac                             [project layer]
                 |
        +--------+-------------+--------+
        |        |             |        |
        |        v             |        |
        |      glozz           |        |       [tool layer]
        |        |             |        |
        v        v             v        v
     corpus -> annotation <- fusion <- graph    [base layer]

Support for other projects would consist in adding writing other project layer
modules that map down to the tool layer.

Departures from the ideal (2013-05-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Educe is still its early stages.  Some departures you may want to be aware of:

* fusion layer does not really exist yet; `educe.annotation` currently
  takes on some of the job (for example, the `text_span` function makes
  annotations of different types more or less comparable)

* layer violations: ideally we want lower layers to be abstract from things
  above them, but you may find eg. glozz-specific assumptions in the base
  layer, which isn't great.
"""
