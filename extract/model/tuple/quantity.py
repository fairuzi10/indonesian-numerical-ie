'''
Quantity exraction class model.
'''

from .predictor import Predictor


class QuantityPredictor(Predictor):

    def build_chunk(self):
        labels = {'NP'}
        grammar = '''
            NP: {<NN|NNP>(<CC|FW|IN|JJ|MD|NEG|NN|NND|NNP|PR|PRP|RB|RP|SC|UH>*<FW|JJ|NN|NNP|NND|PR|PRP>)?|<JJ>}
        '''
        self.build_chunk_with_labels(labels, grammar)

    def get_pred(self):
        for now in self.num_span:
            for child, rel in self.children[now]:
                if child not in self.num_span and rel in {'nmod', 'compound', 'case'}\
                        and self.chunk[child] and not self.chunk[child].used:
                    self.chunk[child].used = True
                    return " ".join(self.chunk[child].words)

            now = self.parent[now][0]

        node, rel = self.parent[self.root_num]
        if node is None:
            return None
        if self.chunk[node]:
            self.chunk[node].used = True
            return " ".join(self.chunk[node].words)
        else:
            return self.words[node]
