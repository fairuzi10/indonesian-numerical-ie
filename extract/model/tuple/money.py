'''
Money exraction class model.
'''

from .cardinal import CardinalPredictor


# The pred extractor is similar to cardinal but without prefix `jumlah`
class MoneyPredictor(CardinalPredictor):

    def get_pred(self):
        node, rel = self.parent[self.root_num]
        if node is None:
            return None
        elif self.chunk[node]:
            self.chunk[node].used = True
            return " ".join(self.chunk[node].words)
        else:
            return self.words[node]
