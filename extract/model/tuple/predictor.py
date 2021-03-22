'''
Relation exraction base class model.
'''

import stanza
import nltk
stanza.download('id')

nlp = stanza.Pipeline(
    lang='id', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)


class Predictor:
    class Chunk:
        def __init__(self, label, words, indexes, dist_from_num, min_level):
            self.label = label
            self.words = words
            self.indexes = indexes
            self.dist_from_num = dist_from_num
            self.min_level = min_level
            self.used = False

        def __str__(self):
            return f'label: {self.label}; words: {self.words}; indexes: {self.indexes}; used: {self.used}'

    def __init__(self, sent, num_span, pos_tagger):
        self.sent = sent
        self.num_span = num_span
        self.pos_tagger = pos_tagger
        pos_tags_only = self.pos_tagger.tag(sent)
        self.words = self.pos_tagger.sent_tokenized
        self.pos_tags = [(word, pos_tag)
                         for word, pos_tag in zip(self.words, pos_tags_only)]
        self.doc = nlp([self.words])
        self.nodes = self.doc.sentences[0].words
        self.build_edges()
        self.build_tree()
        self.build_chunk()

    def build_edges(self):
        self.children = [[] for i in range(len(self.nodes))]
        self.parent = [(None, None) for i in range(len(self.nodes))]
        for word in self.nodes:
            if word.head > 0:
                self.children[word.head - 1].append((word.id - 1, word.deprel))
                self.parent[word.id - 1] = (word.head - 1, word.deprel)

    def build_tree(self):
        self.root = -1
        for word in self.nodes:
            if word.deprel == "root":
                self.root = word.id - 1
        self.level = [-1 for i in range(len(self.nodes))]
        self.level[self.root] = 0
        self.traverse(self.root)

        self.root_num = self.num_span[0]
        for num_idx in self.num_span:
            if self.level[num_idx] < self.level[self.root_num]:
                self.root_num = num_idx

        self.leaf_num = self.num_span[0]
        for num_idx in self.num_span:
            if self.level[num_idx] > self.level[self.leaf_num] or \
                (self.level[num_idx] == self.level[self.leaf_num] and num_idx > self.leaf_num):
                self.leaf_num = num_idx

    def traverse(self, u):
        for v, rel in self.children[u]:
            self.level[v] = self.level[u] + 1
            self.traverse(v)

    def generate_chunk(self, label, words, indexes):
        dist_from_num = max(abs(self.num_span[0]-indexes[-1]), abs(indexes[0]-self.num_span[-1]))
        min_level = min([self.level[idx] for idx in indexes])
        return Predictor.Chunk(label, words, indexes, dist_from_num, min_level)

    def build_chunk_with_labels(self, labels, grammar):
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(self.pos_tags)
        idx = 0
        self.chunks = []
        self.chunk = [None for i in range(len(self.nodes))]
        for subtree in tree.subtrees():
            if subtree.label() in labels:
                chunk_indexes = []
                for i in range(idx, len(self.pos_tags) - len(subtree) + 1):
                    match = True
                    for j in range(len(subtree)):
                        if subtree[j] != self.pos_tags[i+j]:
                            match = False
                            break

                    if match:
                        chunk_indexes = list(range(i, i+len(subtree)))
                        idx = i+len(subtree)
                        break

                self.chunks.append(self.generate_chunk(subtree.label(), [token[0] for token in subtree], chunk_indexes))
                for idx in chunk_indexes:
                    self.chunk[idx] = self.chunks[-1]

    def build_chunk(self):
        raise NotImplementedError()

    def get_pred(self):
        raise NotImplementedError()

    def get_arg0(self):
        node_rel = self.parent[self.root_num]

        while node_rel != (None, None):
            now = node_rel[0]
            for child_rel in self.children[now]:
                child, rel = child_rel
                if rel[:5] == 'nsubj':
                    if self.chunk[child] and not self.chunk[child].used:
                        self.chunk[child].used = True
                        return " ".join(self.chunk[child].words)
            node_rel = self.parent[now]

        return None

    def get_arg1(self):
        return " ".join(self.words[self.num_span[0]:self.num_span[-1]+1])

    def get_tuple(self):
        pred = self.get_pred()
        arg0 = self.get_arg0()
        if not arg0 or not pred:
            arg0 = None
            pred = None
        return (arg0, pred, self.get_arg1())
