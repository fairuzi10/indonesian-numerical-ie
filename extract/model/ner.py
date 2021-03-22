'''
NER tagger class model.
It uses `model_ner` which is the result of NER training using notebook.
'''
import pickle
import sklearn_crfsuite
import os

MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_ner')
NUM_LABELS = ('MON', 'PRC', 'CRD', 'QTY')

class NERTagger():
    NUMERIC = ('nol', 'satu', 'dua', 'tiga', 'empat',
               'lima', 'enam', 'tujuh', 'delapan', 'sembilan')

    def __init__(self, pos_tagger):
        with open(MODEL_FILE_PATH, 'rb') as model:
            self.model = pickle.load(model)
        self.pos_tagger = pos_tagger

    def tag(self, sent):
        self.pos_tags = self.pos_tagger.tag(sent)
        self.sent_tokenized = self.pos_tagger.sent_tokenized
        feature_list = self.sent2features(
            [(word, pos_tag) for word, pos_tag in zip(self.sent_tokenized, self.pos_tags)])
        predicted_tags = self.model.predict_single(feature_list)
        predicted_num_tags = [tag if tag[2:]
                              in NUM_LABELS else 'O' for tag in predicted_tags]
        return predicted_num_tags

    def word2features(self, sent, i):
        token = sent[i][0]
        postag = sent[i][1]

        features = {
            'token.lower()': token.lower(),
            'token.istitle()': token.istitle(),
            'token.isupper()': token.isupper(),
            'token.postag()': postag,
            'token.isnumeric()': token.replace('.', '').replace('-', '').isdigit() or (token.lower() in self.NUMERIC),
            'token.is4digit()': len(token) == 4 and sum([1 if char.isdigit() else 0 for char in token]) == 4
        }
        if i > 0:
            token1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:token.lower()': token1.lower(),
                '-1:token.istitle()': token1.istitle(),
                '-1:token.isupper()': token1.isupper(),
                '-1:token.postag()': postag1,
            })
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            token1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:token.lower()': token1.lower(),
                '+1:token.istitle()': token1.istitle(),
                '+1:token.isupper()': token1.isupper(),
                '+1:token.postag()': postag1,
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]
