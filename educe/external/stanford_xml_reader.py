"""
Reader for Stanford CoreNLP pipeline outputs

Example of output:

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


import sys, os
import operator


try:
    import xml.etree.cElementTree as ET # python 2.5 and later
except ImportError:
    import cElementTree as ET
except ImportError:
    raise ImportError("cElementTree missing!")


class Preprocessing_Source( object ):
    ''' Reads in document annotations produced by CoreNLP pipeline '''
    def __init__( self, encoding="utf-8" ):
        self._encoding = encoding
        return


    def read( self, base_file, suffix=".raw.stanford" ):
        # init annotations
        self._doc_id = os.path.basename(base_file)
        self._sentences = {} # include parse and basic dependencies
        self._tokens = {} # include word form, lemma, pos, and NE tag
        self._offset2sentence = {} # NB: not inclusive
        self._offset2token = {}
        # self._coref_links = {} ## TODO
        # parse -------------------------------------

        parser = ET.XMLParser( encoding=self._encoding )
        file2parse = base_file + suffix
        try:
            root = ET.parse( file2parse )
        except SyntaxError: # Hack around fact that stanford outputs
                            # are not always proper UTF8
            os.system("iconv -f ISO-8859-1 -t UTF-8 %s > tmp; mv tmp %s" \
                      %(file2parse,file2parse))
            root = ET.parse( file2parse )
        """ register sentences """
        s_elts = root.findall(".//sentences/sentence")
        for s in s_elts:
            sid = s.get('id')
            assert sid != None
            # sentence dictionary
            s_dict = dict(id = sid)
            """ register tokens """
            token_list = []
            t_elts = s.findall(".//token")
            for t in t_elts:
                tid = t.get('id')
                assert tid != None
                # token dictionary with basic attributes
                t_start = int(t.find("CharacterOffsetBegin").text)
                t_end = int(t.find("CharacterOffsetEnd").text) - 1 # NB: not inclusive
                t_dict = dict(id = sid+"-"+tid, # original token ID not unique
                              extent = (t_start, t_end),
                              word = xml_unescape(t.find("word").text),
                              s_id = sid) # pointer to sentence ID
                # additional token annotations
                try:
                    t_dict.update(POS = xml_unescape(t.find("POS").text))
                except AttributeError:
                    t_dict.update(POS = None)
                try:
                    t_dict.update(lemma = xml_unescape(t.find("lemma").text))
                except AttributeError:
                    t_dict.update(lemma = None)
                try:
                    t_dict.update(NER = xml_unescape(t.find("NER").text))
                except AttributeError:
                    t_dict.update(NER = None)
                # store token annotation
                self._tokens[(sid,tid)] = t_dict
                token_list.append( t_dict )
                # update token offset maps
                for pos in xrange(t_start,t_end+1):
                    assert pos not in self._offset2token
                    self._offset2token[pos] = t_dict
            token_list.sort( key=lambda x:x['extent'] )
            # update sentence dictionary based on token list
            s_start = token_list[0]['extent'][0]
            s_end = token_list[-1]['extent'][1]
            s_dict.update(extent = (s_start, s_end),
                          tokens = [t['id'] for t in token_list]) # pointer to token ID list
            """ register parse """
            try:
                s_dict.update(parse = xml_unescape(s.find("parse").text))
            except AttributeError:
                s_dict.update(parse = None)

            """ register dependencies """ # FIXME: distinguish basic/collapsed dependencies!
            d_elts = s.findall(".//dep")
            dep_triples = []
            for d in d_elts:
                d_rel = d.get("type")
                gov_id = d.find("governor").get("idx")
                dep_id = d.find("dependent").get("idx")
                dep_triples.append( (d_rel, gov_id, dep_id) )
            s_dict.update( dependencies = dep_triples )

            # store sentence annotation
            self._sentences[sid] = s_dict
            # update sentence offset map
            for pos in xrange(s_start,s_end+1):
                assert pos not in self._offset2sentence
                self._offset2sentence[pos] = s_dict
        return


    def get_document_id( self ):
        doc_id = self._doc_id
        assert doc_id != None and doc_id != "", "Stanford XML reader error: document ID is None or empty string!"
        return self._doc_id


    def get_sentence_annotations(self):
        return self._sentences


    def get_ordered_sentence_list(self, sort_attr="extent"):
        sentences = self.get_sentence_annotations().values()
        return sorted( sentences, key=operator.itemgetter(sort_attr) )


    def get_token_annotations(self):
        return self._tokens


    def get_ordered_token_list(self, sort_attr="extent"):
        tokens = self.get_token_annotations().values()
        return sorted( tokens, key=operator.itemgetter(sort_attr) )


    def get_offset2sentence_map(self):
        return self._offset2sentence


    def get_offset2token_maps(self):
        return self._offset2token





def test_file( base_filename, suffix=".raw.stanford" ):
    reader = Preprocessing_Source()
    reader.read( base_filename, suffix=suffix )
    sentences = reader.get_sentence_annotations().values()
    sentences.sort(key=lambda x:x['extent'])
    tokens = reader.get_token_annotations().values()
    for s in sentences:
        print s['id'], s['extent'], s
    #tokens.sort(key=lambda x:x['extent'])
    #print "\n>> TOKENS:", tokens
    return




if __name__ == "__main__":
    import sys, os
    arg = sys.argv[1]
    suffix = ".xml"
    if not arg.endswith( suffix ):
        sys.exit("File needs to be suffixed by '%s'" %suffix)
    base_name = arg[:-len(suffix)]
    test_file( base_name, suffix=suffix )
