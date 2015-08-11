"""
Reader for Stanford CoreNLP pipeline outputs

Example of output:

.. code-block:: xml

  <document>
    <sentences>
      <sentence id="1">
        <tokens>
        ...
        <token id="19">
        <word>direction</word>
        <lemma>direction</lemma>
        <CharacterOffsetBegin>135</CharacterOffsetBegin>
        <CharacterOffsetEnd>144</CharacterOffsetEnd>
        <POS>NN</POS>
        </token>
        <token id="20">
        <word>.</word>
        <lemma>.</lemma>
        <CharacterOffsetBegin>144</CharacterOffsetBegin>
        <CharacterOffsetEnd>145</CharacterOffsetEnd>
        <POS>.</POS>
        </token>
        ...
        <parse>(ROOT (S (PP (IN For) (NP (NP (DT a) (NN look)) (PP (IN at) (SBAR (WHNP (WP what)) (S (VP (MD might) (VP (VB lie) (ADVP (RB ahead)) (PP (IN for) (NP (NNP U.S.) (NNS forces)))))))))) (, ,) (VP (VB let) (S (NP (POS 's)) (VP (VB turn) (PP (TO to) (NP (NP (PRP$ our) (NNP Miles) (NNP O'Brien)) (PP (IN in) (NP (NNP Atlanta)))))))) (. .))) </parse>
        <basic-dependencies>
          <dep type="prep">
            <governor idx="13">let</governor>
            <dependent idx="1">For</dependent>
          </dep>
          ...
        </basic-dependencies>
        <collapsed-dependencies>
          <dep type="det">
            <governor idx="3">look</governor>
            <dependent idx="2">a</dependent>
          </dep>
          ...
        </collapsed-dependencies>
        <collapsed-ccprocessed-dependencies>
          <dep type="det">
            <governor idx="3">look</governor>
            <dependent idx="2">a</dependent>
          </dep>
          ...
        </collapsed-ccprocessed-dependencies>
      </sentence>
    </sentences>
  </document>


IMPORTANT: Note that Stanford pipeline uses RHS inclusive offsets.

"""

from __future__ import print_function
import sys
import os
import operator


try:
    import xml.etree.cElementTree as ET  # python 2.5 and later
except ImportError:
    import cElementTree as ET
except ImportError:
    raise ImportError("cElementTree missing!")


def xml_unescape(_str):
    """Get a proper string where special XML characters are unescaped.

    Notes
    -----
    You can also use `xml.sax.saxutils.escape`
    """
    # Caution: you have to escape '&' first!
    _str = _str.replace(u'&amp;', u'&')
    _str = _str.replace(u'&lt;', u'<')
    _str = _str.replace(u'&gt;', u'>')
    return _str


class PreprocessingSource(object):
    """Reads in document annotations produced by CoreNLP pipeline.

    This works as a stateful object that stores and provides access to
    all annotations contained in a CoreNLP output file, once the `read`
    method has been called.
    """

    def __init__(self, encoding="utf-8"):
        self._encoding = encoding
        return

    def read(self, base_file, suffix=".raw.stanford"):
        """Read and store the annotations from CoreNLP's output.

        This function does not return anything, it modifies the state
        of the object to store the annotations.
        """
        # init annotations
        self._doc_id = os.path.basename(base_file)
        self._sentences = {}  # include parse and basic dependencies
        self._tokens = {}  # include word form, lemma, pos, and NE tag
        self._offset2sentence = {}  # NB: not inclusive
        self._offset2token = {}
        self._coref_chains = []  # from sentence to chain

        # parse -------------------------------------
        parser = ET.XMLParser(encoding=self._encoding)
        file2parse = base_file + suffix
        root = ET.parse(file2parse)  # MM missing arg: ", parser=parser" ?

        # register sentences
        s_elts = root.findall(".//sentences/sentence")
        for s in s_elts:
            sid = s.get('id')
            assert sid is not None
            # sentence dictionary
            s_dict = dict(id=sid)

            # register tokens
            token_list = []
            t_elts = s.findall(".//token")
            for t in t_elts:
                tid = t.get('id')
                assert tid is not None
                # token dictionary with basic attributes
                t_start = int(t.find("CharacterOffsetBegin").text)
                # NB: not inclusive
                t_end = int(t.find("CharacterOffsetEnd").text) - 1
                # original token ID not unique
                # s_id: pointer to sentence ID
                t_dict = dict(id=self._mk_token_id(sid)(tid),
                              extent=(t_start, t_end),
                              word=xml_unescape(t.find("word").text),
                              s_id=sid)
                # additional token annotations
                try:
                    t_dict.update(POS=xml_unescape(t.find("POS").text))
                except AttributeError:
                    t_dict.update(POS=None)
                try:
                    t_dict.update(lemma=xml_unescape(t.find("lemma").text))
                except AttributeError:
                    t_dict.update(lemma=None)
                try:
                    t_dict.update(NER=xml_unescape(t.find("NER").text))
                except AttributeError:
                    t_dict.update(NER=None)
                # store token annotation
                self._tokens[(sid, tid)] = t_dict
                token_list.append(t_dict)
                # update token offset maps
                for pos in xrange(t_start, t_end+1):
                    assert pos not in self._offset2token
                    self._offset2token[pos] = t_dict
            token_list.sort(key=lambda x: x['extent'])
            # update sentence dictionary based on token list
            s_start = token_list[0]['extent'][0]
            s_end = token_list[-1]['extent'][1]
            # tokens: pointer to token ID list
            s_dict.update(extent=(s_start, s_end),
                          tokens=[t['id'] for t in token_list])

            # register parse
            try:
                s_dict.update(parse=xml_unescape(s.find("parse").text))
            except AttributeError:
                s_dict.update(parse=None)

            # register dependencies
            basic_deps = self._read_deps(sid, s, 'basic-dependencies')
            colla_deps = self._read_deps(sid, s, 'collapsed-dependencies')
            co_cc_deps = self._read_deps(sid, s,
                                         'collapsed-ccprocessed-dependencies')
            s_dict.update(
                dependencies=basic_deps,
                collapsed_dependencies=colla_deps,
                collapsed_cc_dependencies=co_cc_deps)

            # store sentence annotation
            self._sentences[sid] = s_dict
            # update sentence offset map
            for pos in xrange(s_start, s_end+1):
                assert pos not in self._offset2sentence
                self._offset2sentence[pos] = s_dict

        # register coreference chains
        coref_elts = root.findall('.//coreference/coreference')
        self._coref_chains = [self._read_chain(x) for x in coref_elts]

        return

    def _mk_token_id(self, sid):
        """Get token ids relative to a sentence id"""
        return lambda x: sid + '-' + x

    def _read_deps(self, sid, xml, dep_type):
        """Read dependencies from an XML element

        Parameters
        ----------
        sid: string
            Sentence id
        xml: XML element
            XML element
        dep_type: string
            Type of the (Stanford) dependencies
        """
        xpath = ".//dependencies[@type='%s']/dep" % dep_type
        dep_triples = []
        mk_id = self._mk_token_id(sid)
        for d in xml.findall(xpath):
            d_rel = d.get("type")
            gov_id = d.find("governor").get("idx")
            dep_id = d.find("dependent").get("idx")
            dep_triples.append((d_rel, mk_id(gov_id), mk_id(dep_id)))
        return dep_triples

    def _read_chain(self, xml):
        """Read a chain of coreferences.

        Parameters
        ----------
        xml: XML element
        """
        xpath = './mention'
        return [self._read_mentions(x) for x in xml.findall(xpath)]

    def _read_mentions(self, xml):
        """Read mentions of a chain of coreferences"""
        sid = xml.find('sentence').text.strip()
        mk_id = self._mk_token_id(sid)

        def get_id(name):
            """Get id for name"""
            return mk_id(xml.find(name).text.strip())

        representative = xml.get('representative', '').strip() == 'true'
        return {
            'start': get_id('start'),
            'end': get_id('end'),
            'head': get_id('head'),
            'sentence': sid,
            'most_representative': representative,
            }

    def get_document_id(self):
        """Get the document id"""
        doc_id = self._doc_id
        if doc_id is None or doc_id == "":
            raise ValueError('Stanford XML reader error:',
                             ' document ID is None or empty string!')
        return self._doc_id

    def get_sentence_annotations(self):
        """Get the annotations of all sentences"""
        return self._sentences

    def get_coref_chains(self):
        """Get all coreference chains"""
        return self._coref_chains

    def get_ordered_sentence_list(self, sort_attr="extent"):
        """Get the list of sentences, ordered by sort_attr"""
        sentences = self.get_sentence_annotations().values()
        return sorted(sentences, key=operator.itemgetter(sort_attr))

    def get_token_annotations(self):
        """Get the annotations of all tokens"""
        return self._tokens

    def get_ordered_token_list(self, sort_attr="extent"):
        """Get the list of tokens, ordered by sort_attr"""
        tokens = self.get_token_annotations().values()
        return sorted(tokens, key=operator.itemgetter(sort_attr))

    def get_offset2sentence_map(self):
        """Get the offset to each sentence"""
        return self._offset2sentence

    def get_offset2token_maps(self):
        """Get the offset to each token"""
        return self._offset2token


def test_file(base_filename, suffix=".raw.stanford"):
    """Test that a file is effectively readable and print sentences"""
    reader = PreprocessingSource()
    reader.read(base_filename, suffix=suffix)

    sentences = reader.get_sentence_annotations().values()
    sentences.sort(key=lambda x: x['extent'])
    for s in sentences:
        print(s['id'], s['extent'], s)

    tokens = reader.get_token_annotations().values()
    # tokens.sort(key=lambda x:x['extent'])
    # print("\n>> TOKENS:", tokens)
    return


if __name__ == "__main__":
    arg = sys.argv[1]
    suffix = ".xml"
    if not arg.endswith(suffix):
        sys.exit("File needs to be suffixed by '%s'" % suffix)
    base_name = arg[:-len(suffix)]
    test_file(base_name, suffix=suffix)
