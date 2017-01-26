""" Fake graphs for testing STAC algorithms

Specification for mini-language

Source string is parsed line by line, data type depends on first character
Uppercase letters are speakers, lowercase letters are units
EDU names are arranged following alphabetical order (does NOT apply to CDUs)
Please arrange the lines in that order:

* # : speaker line ::

     # Aabce Bdg Cfh

* any lowercase : CDU line (top-level last) ::

     y(eg) x(wyz)

* S or C : relation line ::

     Sabd bf ceCh

anything else : skip as comment
"""

from __future__ import print_function

from collections import defaultdict
import re
import string
import textwrap


from educe.annotation import (Span, RelSpan,
                              Unit, Relation, Schema,
                              Document)
from educe.corpus import FileId
from educe.stac.graph import Graph, DotGraph
# from educe.graph import Graph, DotGraph
from educe.stac.util.output import write_dot_graph


class LightGraph:
    """ Structure holding only relevant information

    Unit keys (sortable, hashable) must correspond to reading order
    CDUs can be placed in any position wrt their components
    """
    def __init__(self, src):
        """ Empty graph """
        self.speakers = defaultdict(set)
        self.info = defaultdict(lambda: None)
        self.down = defaultdict(list)
        self.cdus = defaultdict(set)
        self.anno_map = dict()

        self._load_txt(src)
        self.doc = self._mk_doc()

    def _load_txt(self, src):
        """ Load graph from mini-language """
        lines = re.split(r'[\r\n\/]+', src)
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith('#'):
                # Speaker line
                blocks = re.split(r'\s+', line[1:])
                for block in blocks:
                    if not block:
                        continue
                    speaker = block[0]
                    for unit in block[1:]:
                        self.speakers[unit].add(speaker)
            elif line.startswith(('S', 'C')):
                # Relation line
                mode = line[0]
                blocks = re.split(r'\s+', line[1:])
                for block in blocks:
                    last = None
                    for letter in block:
                        if letter in ('S', 'C'):
                            mode = letter
                        else:
                            if last is not None:
                                self.down[last].append((letter, mode))
                            last = letter
            elif re.match(r'[a-z]', line):
                # CDU line
                # TODO
                blocks = re.split(r'\s+', line)
                for block in blocks:
                    unit = block[0]
                    for sub in block[2:-1]:
                        self.cdus[unit].add(sub)
                        self.speakers[unit] |= self.speakers[sub]
            else:
                continue

    def _mk_doc(self):
        """ Create an educe.annotation.Document from this graph """
        def start(name):
            return ord(name) - ord('a')

        def glozz_id(name):
            return 'du_' + str(start(name))

        def is_edu(name):
            return name not in self.cdus

        anno_units = list()
        anno_cdus = list()
        anno_rels = list()

        for du_name, speaker_set in self.speakers.items():
            # EDU loop
            if not is_edu(du_name):
                continue

            du_start, du_glozz_id = start(du_name), glozz_id(du_name)
            x_edu = Unit(du_glozz_id,
                         Span(du_start, du_start+1),
                         'Segment',
                         dict())
            speaker = list(speaker_set)[0]
            turn = Unit('t' + du_glozz_id,
                        Span(du_start, du_start+1),
                        'Turn',
                        {'Identifier': du_start, 'Emitter': speaker})

            self.anno_map[du_name] = x_edu
            anno_units.append(x_edu)
            anno_units.append(turn)

        for du_name, sub_names in self.cdus.items():
            x_cdu = Schema(glozz_id(du_name),
                           set(glozz_id(x)
                               for x in sub_names if is_edu(x)),
                           set(),
                           set(glozz_id(x)
                               for x in sub_names if not is_edu(x)),
                           'Complex_discourse_unit',
                           dict())
            self.anno_map[du_name] = x_cdu
            anno_cdus.append(x_cdu)

        rel_count = 0
        for src_name in self.down:
            for tgt_name, rel_tag in self.down[src_name]:
                rel_glozz_id = 'rel_'+str(rel_count)
                rel_count += 1
                if rel_tag == 'S':
                    rel_name = 'Q-Elab'
                elif rel_tag == 'C':
                    rel_name = 'Contrast'
                else:
                    raise ValueError('Unknown tag {0}'.format(rel_tag))

                rel = Relation(rel_glozz_id,
                               RelSpan(glozz_id(src_name),
                                       glozz_id(tgt_name)),
                               rel_name,
                               dict())
                self.anno_map[(src_name, tgt_name)] = rel
                anno_rels.append(rel)

        dialogue = Unit('dialogue_0',
                        Span(0, max(u.text_span().char_end
                                    for u in anno_units)),
                        'Dialogue',
                        {})
        anno_units.append(dialogue)

        doc = Document(anno_units, anno_rels, anno_cdus,
                       string.ascii_lowercase)
        return doc

    def get_doc(self):
        return self.doc

    def get_node(self, name):
        """ Return an educe.annotation.Unit or Schema
            for the given LightGraph name
        """
        return self.anno_map[name]

    def get_edge(self, source, target):
        """ Return an educe.annotation.Relation
            for the given LightGraph names for source and target
        """
        return self.anno_map[(source, target)]


if __name__ == '__main__':
    src = textwrap.dedent("""
    == Test
    # Aae Bb Ccf Dd
    Sabe ace ade ef
    """)

    src2 = textwrap.dedent("""
    == Test
    # Aabc Bdef
    x(def) y(xc)
    Sabc Cdef Cay
    """)
    lg = LightGraph(src)
    doc = lg.get_doc()
    doc_id = FileId('test', '01', 'discourse', 'GOLD')
    doc.set_origin(doc_id)
    graph = Graph.from_doc({doc_id: doc}, doc_id)

    dot_graph = DotGraph(graph)
    write_dot_graph(doc_id, '/tmp/graph', dot_graph)
