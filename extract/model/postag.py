'''
POS tagger class model.
It uses `model_postag` which is the result of POS tag training using notebook.
'''
import os
import nltk
from nltk.tokenize import word_tokenize
import pickle
nltk.download('punkt')

MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_postag')

class POSTagger():
    def __init__(self):
        with open(MODEL_FILE_PATH, 'rb') as model:
            self.model = pickle.load(model)

    def tag(self, sent):
        if isinstance(sent, list):
            self.sent_tokenized = sent
        elif isinstance(sent, str):
            self.sent_tokenized = word_tokenize(sent)
        for i in range(len(self.sent_tokenized)):
            if self.sent_tokenized[i] == '``' or self.sent_tokenized[i] == "''":
                self.sent_tokenized[i] = '"'
        self.pos_tags = self.model.predict_single(
            self.word2features(self.sent_tokenized))
        return self.pos_tags

    def word2features(self, sent):
        feature_list = []
        for i in range(len(sent)):
            token = sent[i]
            features = {
                'token': token,
                'token.lower()': token.lower(),
                'token.istitle()': token.istitle(),
                'token.isupper()': token.isupper(),
                'token[:2]': token[:2],
                'token[:3]': token[:3],
                'token[-3:]': token[:-3],
                'token[-2:]': token[:-2],
            }
            if i > 0:
                token1 = sent[i-1]
                features.update({
                    '-1:token.lower()': token1.lower(),
                    '-1:token.istitle()': token1.istitle(),
                    '-1:token.isupper()': token1.isupper(),
                })
            else:
                features['BOS'] = True
            if i < len(sent)-1:
                token1 = sent[i+1]
                features.update({
                    '+1:token.lower()': token1.lower(),
                    '+1:token.istitle()': token1.istitle(),
                    '+1:token.isupper()': token1.isupper(),
                })
            else:
                features['EOS'] = True

            feature_list.append(features)

        return feature_list
