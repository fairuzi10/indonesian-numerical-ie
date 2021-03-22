'''
Cardinal exraction class model.
'''

from .predictor import Predictor


class CardinalPredictor(Predictor):

    def build_chunk(self):
        labels = {'NP'}
        grammar = '''
            NP: {<NN|NNP>(<CC|FW|IN|JJ|MD|NEG|NN|NND|NNP|PR|PRP|RB|RP|SC|UH>*<FW|JJ|NN|NNP|NND|PR|PRP>)?}
        '''
        self.build_chunk_with_labels(labels, grammar)

    def get_pred(self):
        node, rel = self.parent[self.root_num]
        if node is None:
            return None
        elif self.chunk[node]:
            self.chunk[node].used = True
            return "jumlah " + " ".join(self.chunk[node].words)
        else:
            return "jumlah " + self.words[node]
