'''
Percent exraction class model.
'''

from .predictor import Predictor


class PercentPredictor(Predictor):

    def build_chunk(self):
        labels = {'NP'}
        grammar = '''
            NP: {<NN|NNP>(<CC|FW|IN|JJ|MD|NEG|NN|NND|NNP|PR|PRP|RB|RP|SC|UH>*<FW|JJ|NN|NNP|NND|PR|PRP>)?}
        '''
        self.build_chunk_with_labels(labels, grammar)

    def get_pred(self):
        node_rel = self.parent[self.root_num]

        while node_rel != (None, None):
            now = node_rel[0]
            for child_rel in self.children[now]:
                child, rel = child_rel
                if rel == 'nmod' or rel == 'appos':
                    if self.chunk[child] and not self.chunk[child].used:
                        self.chunk[child].used = True
                        return "persentase " + " ".join(self.chunk[child].words)
            node_rel = self.parent[now]

        return None
