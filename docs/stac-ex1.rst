
STAC example: Turns and resources
=================================

Suppose you wanted to find the following (an actual request from the
STAC project)

“Player offers to give resource X (possibly for Y) but does not hold
resource X.”

In this tutorial, we'll walk through such a query applying it to a
single file in the corpus. Before digging into the tutorial proper,
let's first read the sample data.

.. code:: python

    from __future__ import print_function
    from educe.corpus import FileId
    import educe.stac
    
    # relative to the educe docs directory
    data_dir = '../data'
    corpus_dir = '{dd}/stac-sample'.format(dd=data_dir)
    
    def text_snippet(text):
        "short text fragment"
        if len(text) < 43:
            return text
        else:
            return "{0}...{1}".format(text[:20], text[-20:])
        
    def preview_unit(doc, anno):
        "the default str(anno) can be a bit overwhelming"
        preview = "{span: <11} {id: <20} [{type: <12}] {text}"
        text = doc.text(anno.text_span())
        return preview.format(id=anno.local_id(),
                              type=anno.type,
                              span=anno.text_span(),
                              text=text_snippet(text))
    
    # pick out an example document to work with creating FileIds by hand
    # is not something we would typically do (normally we would just iterate
    # through a corpus), but it's useful for illustration
    ex_key = FileId(doc='s1-league2-game3',
                    subdoc='03',
                    stage='units',
                    annotator='BRONZE')
    reader = educe.stac.Reader(corpus_dir)
    ex_files = reader.filter(reader.files(),
                               lambda k: k == ex_key)
    corpus = reader.slurp(ex_files, verbose=True)
    ex_doc = corpus[ex_key]


.. parsed-literal::

    Slurping corpus dir [1/1 done]


1. Turn and resource annotations
--------------------------------

How would you go about doing it? One place to start is to look at turns
and resources independently. We can filter turns and resources with the
helper functions ``is_turn`` and ``is_resource`` from ``educe.stac``

.. code:: python

    import educe.stac
    
    ex_turns = [x for x in ex_doc.units if educe.stac.is_turn(x)]
    ex_offers = [x for x in ex_doc.units if educe.stac.is_resource(x)
                 and x.features['Status'] == 'Givable']
    
    print("Example turns")
    print("-------------")
    for anno in ex_turns[:5]:
        # notice here that unit annotations have a features field
        print(preview_unit(ex_doc, anno))
    
    print()
    print("Example resources")
    print("-----------------")
    for anno in ex_offers[:5]:
        # notice here that unit annotations have a features field
        print(preview_unit(ex_doc, anno))
        print('', anno.features)


.. parsed-literal::

    Example turns
    -------------
    (35,66)     stac_1368693098      [Turn        ] 152 : sabercat : yep, for what?
    (100,123)   stac_1368693104      [Turn        ] 154 : sabercat : no way
    (146,171)   stac_1368693110      [Turn        ] 156 : sabercat : could be
    (172,191)   stac_1368693113      [Turn        ] 157 : amycharl : :)
    (192,210)   stac_1368693116      [Turn        ] 160 : amycharl : ?
    
    Example resources
    -----------------
    (84,88)     asoubeille_1374939917916 [Resource    ] clay
     {'Status': 'Givable', 'Kind': 'clay', 'Correctness': 'True', 'Quantity': '?'}
    (141,144)   asoubeille_1374940096296 [Resource    ] ore
     {'Status': 'Givable', 'Kind': 'ore', 'Correctness': 'True', 'Quantity': '?'}
    (398,403)   asoubeille_1374940373466 [Resource    ] sheep
     {'Status': 'Givable', 'Kind': 'sheep', 'Correctness': 'True', 'Quantity': '?'}
    (464,467)   asoubeille_1374940434888 [Resource    ] ore
     {'Status': 'Givable', 'Kind': 'ore', 'Correctness': 'True', 'Quantity': '1'}
    (689,692)   asoubeille_1374940671003 [Resource    ] one
     {'Status': 'Givable', 'Kind': 'Anaphoric', 'Correctness': 'True', 'Quantity': '1'}


Oh no, Anaphors
~~~~~~~~~~~~~~~

Oh dear, some of our resources won't tell us their types directly. They
are anaphors pointing to other annotations. We'll ignore these for the
moment, but it'll be important to deal with them properly later on.

2. Resources within turns?
--------------------------

| It's not enough to be able to spit out resource and turn annotations.
| What we really want to know about are which resources are within which
turns'

.. code:: python

    ex_turns_with_offers = [t for t in ex_turns if any(t.encloses(r) for r in ex_offers)]
    
    print("Turns and resources within")
    print("--------------------------")
    for turn in ex_turns_with_offers[:5]:
        t_resources = [x for x in ex_resources if turn.encloses(x)]
        print(preview_unit(ex_doc, turn))
        for rsrc in t_resources:
            kind = rsrc.features['Kind']
            print("\t".join(["", str(rsrc.text_span()), kind]))

3. But does the player own these resources?
-------------------------------------------

Now that we can extract the resources within a turn, our next task is to
figure out if the player actually has these resources to give. This
information is stored in the turn features.

.. code:: python

    def parse_turn_resources(turn):
        """Return a dictionary of resource names to counts thereof
        """
        def split_eq(attval):
            key, val = attval.split('=')
            return key.strip(), int(val)
        rxs = turn.features['Resources']
        return dict(split_eq(x) for x in rxs.split(';')) 
    
    print("Turns and player resources")
    print("--------------------------")
    for turn in ex_turns[:5]:
        t_resources = [x for x in ex_resources if turn.encloses(x)]
        print(preview_unit(ex_doc, turn))
        # not to be confused with the resource annotations within the turn
        print('\t', parse_turn_resources(turn))
            

4. Putting it together: is this an honest offer?
------------------------------------------------

.. code:: python

    def is_somewhat_honest(turn, offer):
        """True if the player has the offered resource 
        """
        if offer.features['Status'] != 'Givable':
            raise ValueError('Resource must be givable')
        kind = offer.features['Kind']
        t_rxs = parse_turn_resources(turn)
        return t_rxs.get(kind, 0) > 0
    
    def is_honest(turn, offer):
        """
        True if the player has the offered resource
        at the quantity offered. Undefined for offers that
        do not have a defined quantity
        """
        if offer.features['Status'] != 'Givable':
            raise ValueError('Resource must be givable')
        if offer.features['Quantity'] == '?':
            raise ValueError('Resource must have a known quantity')
        promised = int(offer.features['Quantity'])    
        kind = rsrc.features['Kind']
        t_rxs = parse_turn_resources(turn)
        return t_rxs.get(kind, 0) >= promised
    
    def critique_offer(turn, offer):
        """Return some commentary on an offered resource"""
        kind = offer.features['Kind']
        quantity = offer.features['Quantity']
        honest = 'n/a' if quantity == '?' else is_honest(turn, offer)
        msg = ("\t{offered}/{has} {kind} | "
               "has some: {honestish}, "
               "enough: {honest}")
        return msg.format(kind=kind,
                          offered=quantity,
                          has=player_rxs.get(kind),
                          honestish=is_somewhat_honest(turn, offer),
                          honest=honest)
    
    ex_turns_with_offers = [t for t in ex_turns if any(t.encloses(r) for r in ex_offers)]
    
    print("Turns and offers")
    print("----------------")
    for turn in ex_turns_with_offers[:5]:
        offers = [x for x in ex_offers if turn.encloses(x)]
        print('', preview_unit(ex_doc, turn))
        player_rxs = parse_turn_resources(turn)
        for offer in offers:
            print(critique_offer(turn, offer))


5. What about those anaphors?
-----------------------------

Anaphors are represented with 'Anaphora' relation instances. Relation
instances have a source and target connecting two unit level annotations
(here two resources). The idea here is that the anaphor would be the
source of the relation, and its antecedant is the target. We'll assume
for simplicity that resource anaphora do not form chains.

.. code:: python

    import copy
    
    resource_types = {}
    for anno in ex_doc.relations:
        if anno.type != 'Anaphora':
            continue
        resource_types[anno.source] = anno.target.features['Kind']
    
    print("Turns and offers (anaphors accounted for)")
    print("-----------------------------------------")
    for turn in ex_turns_with_offers[:5]:
        offers = [x for x in ex_offers if turn.encloses(x)]
        print('', preview_unit(ex_doc, turn))
        player_rxs = parse_turn_resources(turn)
        for offer in offers:
            if offer in resource_types:
                kind = resource_types[offer]
                offer = copy.copy(offer)
                offer.features['Kind'] = kind
            print(critique_offer(turn, offer))

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
